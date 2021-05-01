import sys
import os
sys.path.append(os.path.abspath("../"))

"""Entry point to train IRNet"""
import numpy as np 
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from constants import *
from data.chexpert_dataset_irnet import CheXpertDatasetIRNet
from torch.utils.data import DataLoader
from misc import pyutils, torchutils, indexing
from args.train_arg_parser_irnet import TrainArgParserIRNet
import importlib
from augmentations import get_transforms
import wandb

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Model(LightningModule):
    
#     def __init__(self, hparams):
#         super(Model, self).__init__()
#         self.hparams = hparams

transform_config = {
        #'augmentation_scope': 'horizontal_flip',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
        'masks_normalization': 'none',
        #'masks_output_format_type': 'byte',
        'size': 512,
        'size_transform': 'resize'
    }
transform = get_transforms(transform_config)


def main(args):
    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args
    irnet_args = args.irnet_args

    hyperparameters = dict(
        learning_rate = optim_args.lr,
        weight_decay = optim_args.weight_decay,
        num_epochs = optim_args.num_epochs
    )

    wandb.init(config=hyperparameters, project="weakly_supervised_irnet")
    config = wandb.config

    path_index = indexing.PathIndex(radius=10, default_size=(irnet_args.crop_size // 4, irnet_args.crop_size // 4))

    model = getattr(importlib.import_module(irnet_args.network), 'AffinityDisplacementLoss')(
        path_index)
    
    train_dataset = CheXpertDatasetIRNet("train.csv", None, None, transform=transform, indices_from=path_index.src_indices,
        indices_to=path_index.dst_indices)
    
    train_data_loader = DataLoader(train_dataset, batch_size=data_args.batch_size,
                                   shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // data_args.batch_size) * optim_args.num_epochs
    print("LEARNING RATE:", optim_args.lr)
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1 * optim_args.lr, 'weight_decay': optim_args.weight_decay},
        {'params': param_groups[1], 'lr': 10 * optim_args.lr, 'weight_decay': optim_args.weight_decay}
    ], lr=optim_args.lr, weight_decay=optim_args.weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model.cuda(0), device_ids=['cuda:0'])
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()
    best_train_loss = np.float('inf')

    for ep in range(optim_args.num_epochs):

        print('Epoch %d/%d' % (ep + 1, optim_args.num_epochs))
        total_train_loss = 0.0

        for iter, pack in enumerate(train_data_loader):
            # Img is (batch_size x #channels x w x h)
            img = pack['img']
            img = img.to(device)
            
            #pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)
            # print(pos_aff_loss.shape)
            
            bg_pos_labels = pack['aff_bg_pos_labels'].cuda(0, non_blocking=True)
            fg_pos_labels = pack['aff_fg_pos_labels'].cuda(0, non_blocking=True)
            neg_labels = pack['aff_neg_labels'].cuda(0, non_blocking=True)

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(img, True)
            
            bg_pos_aff_loss = 0.0
            fg_pos_aff_loss = 0.0
            pos_aff_loss_backprop = 0.0
            neg_aff_loss_backprop = 0.0
            dp_fg_loss_backprop = 0.0
            dp_bg_loss_backprop = 0.0

            for index, _ in enumerate(LOCALIZATION_TASKS):
                bg_pos_label = bg_pos_labels[:, index, :, :]
                fg_pos_label = fg_pos_labels[:, index, :, :]
                neg_label = neg_labels[:, index, :, :]

                bg_pos_aff_loss += torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
                fg_pos_aff_loss += torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
                pos_aff_loss_backprop += bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
                neg_aff_loss_backprop += torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)
                

                dp_fg_loss_backprop += torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
                dp_bg_loss_backprop += torch.sum(dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            metrics = {'loss1': pos_aff_loss_backprop.item(), 'loss2': neg_aff_loss_backprop.item(),
                           'loss3': dp_fg_loss_backprop.item(), 'loss4': dp_bg_loss_backprop.item()}
            avg_meter.add(metrics)
            wandb.log(metrics)
            total_loss = (pos_aff_loss_backprop + neg_aff_loss_backprop) / 2 + (dp_fg_loss_backprop + dp_bg_loss_backprop) / 2

            total_train_loss += total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 5 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f %.4f' % (total_loss,
                          avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'),
                          avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * data_args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
            else:
                timer.reset_stage()
        
        mean_train_loss = total_train_loss / len(train_data_loader.dataset)
        print('Total Train Loss: %.4f' % (mean_train_loss))
        wandb.log({"Train Loss": mean_train_loss})
        
        if mean_train_loss < best_train_loss:
            best_train_loss = mean_train_loss
            torch.save(model.module.state_dict(), IRNET_MODEL_SAVE_DIR / f"best_ir_model_hdf5_lr_{optim_args.lr}_weight_decay_{optim_args.weight_decay}.pth")
            


if __name__ == '__main__':
    # path_index = indexing.PathIndex(radius=10, default_size=(512 // 4, 512 // 4))

    # train_dataset = CheXpertDatasetIRNet("train.csv", None, None, indices_from=path_index.src_indices,
    #     indices_to=path_index.dst_indices)
    
    # train_data_loader = DataLoader(train_dataset, batch_size=4,
    #                                shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True, drop_last=True)
    


    # for iter, pack in enumerate(train_data_loader):
    #     print(pack['labels'].shape)

    #torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParserIRNet()
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)
