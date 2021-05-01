"""Entry-point script to train models."""
import json
import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from args import TrainArgParser
from constants import *
from data import get_loader
from eval import Evaluator
from optim import ranger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from util.localization_util import localize
from util.model_util import get_probs, save_predictions

from argparse import Namespace

seed_everything(42)

class Model(LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, hparams):
        super(Model, self).__init__()
        # Get nested namespaces.
        self.hparams = hparams
        args = hparams
        if type(args) is dict:
            args = Namespace(**args)
        self.model_args = args.model_args
        self.logger_args = args.logger_args
        self.optim_args = args.optim_args
        self.data_args = args.data_args
        self.transform_args = args.transform_args
        self.save_cams_tasks = None

        model_fn = models.__dict__[self.model_args.model]
        tasks = self.model_args.__dict__[TASKS]
        self.model = model_fn(tasks, self.model_args)

        evaluator = Evaluator()
        self.loss = evaluator.get_loss_fn(loss_fn_name=self.optim_args.loss_fn,
                                          model_uncertainty=self.model_args.model_uncertainty,
                                          mask_uncertain=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        return self.inference_step(batch, batch_nb)

    def validation_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, phase="valid")

    def test_step(self, batch, batch_nb):
        if self.logger_args.save_cams:
            x, y, info_dict, mask = batch
            b, s, c, h, w = x.shape
            for i in range(b):
                imgs_paths = info_dict['img_paths'][i]
                imgs = x[i]
                imgs_dims = info_dict['img_dims'][i]
                num_imgs = len(imgs_dims)
                for j in range(num_imgs):
                    img_path = str(imgs_paths[j])
                    img_name = img_path[img_path.find('patient'):].replace(".jpg",'').replace('/', '_')
                    img_dims = imgs_dims[j]
                    img = imgs[j].unsqueeze(0)
                    localize(self.model, self.model_args, self.logger_args, self.transform_args,
                             img, img_name, img_dims, y[i], self.save_cams_tasks)

        return self.inference_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
       return self.inference_epoch_end(outputs, phase="test")

    def inference_step(self, batch, batch_nb):
        x, y, info_dict, mask = batch
        # Fuse batch size `b` and study length `s`
        b, s, c, h, w = x.shape
        inputs = x.view(-1, c, h, w)
        logits = self.forward(inputs)
        logits = logits.view(b, s, -1)

        # Mask padding to negative infinity
        ignore_where = (mask == 0).unsqueeze(-1)
        ignore_where = ignore_where.repeat(1, 1, logits.size(-1))
        logits = torch.where(ignore_where,
                             torch.full_like(logits, float('-inf')),
                             logits)
        logits, _ = torch.max(logits, 1)

        return {'logits': logits, 'y': y, 'info_dict': info_dict}

    def inference_epoch_end(self, outputs, phase):
        logits = torch.cat([x['logits'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        loss = self.loss(logits, y)
        probs = get_probs(logits, self.model_args.model_uncertainty)
        self.y_test = y
        self.y_pred = probs
        if phase == "test" and self.model_args.ckpt_path is not None:
            save_predictions(outputs, probs, self.model_args, self.logger_args)
        
        # return None
        # aucs = self.get_aucs(y, probs)
        # avg_auc = torch.tensor(np.mean(list(aucs.values())))

        metrics = {'val_loss': loss}
        # aucs.update(metrics)
        metrics.update(
            {'progress_bar': metrics.copy(),
            #  'log': aucs
        })
        return metrics

    def configure_optimizers(self):
        if self.optim_args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), self.optim_args.lr,
                                       momentum=self.optim_args.sgd_momentum,
                                       weight_decay=self.optim_args.weight_decay,
                                       dampening=self.optim_args.sgd_dampening)
        elif self.optim_args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), self.optim_args.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.optim_args.weight_decay)
        elif self.optim_args.optimizer == 'ranger':
            self.optimizer = ranger.Ranger(self.parameters())
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')
        return [self.optimizer]


    def train_dataloader(self):
        # Get train and valid loader objects.
        train_loader = get_loader(phase="train",
                                 data_args=self.data_args,
                                 transform_args=self.transform_args,
                                 is_training=True,
                                 return_info_dict=False)
        return train_loader

    def val_dataloader(self):
        val_loader = get_loader(phase="valid",
                                data_args=self.data_args,
                                transform_args=self.transform_args,
                                is_training=False,
                                return_info_dict=True)
        return val_loader

    def test_dataloader(self):
        # Get train and valid loader objects.
        test_loader = None
        if self.logger_args.save_train_cams:
            print("Generating train CAMs.....")
            test_loader = get_loader(phase="train",
                                    data_args=self.data_args,
                                    transform_args=self.transform_args,
                                    is_training=False,
                                    return_info_dict=True)
        else:
            test_loader = get_loader(phase=self.data_args.phase,
                                    data_args=self.data_args,
                                    transform_args=self.transform_args,
                                    is_training=False,
                                    return_info_dict=True)
        return test_loader

    def get_aucs(self, y, probs):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")
        y = y.cpu().numpy()
        probs = probs.cpu().numpy()

        aucs = {}
        for index, task in enumerate(self.model.tasks):
            if task not in CHEXPERT_COMPETITION_TASKS:
                continue
            auc = roc_auc_score(y[:, index], probs[:, index])
            aucs['AUC_' + task] = auc
        return aucs


def main(hparams):
    model = Model(hparams)
    logger = WandbLogger(name=hparams.logger_args.wandb_run_name,
                         save_dir=str(hparams.logger_args.save_dir),
                         project=hparams.logger_args.wandb_project_name)
    logger.watch(model, log='all')
    
    if hparams.optim_args.metric_name == 'chexpert_competition_AUROC':
        mode = 'max'
    elif hparams.optim_args.metric_name == 'val_loss':
        mode = 'min'
    filepath = hparams.logger_args.save_dir / \
               logger.version / \
               f'{{epoch}}-{{{hparams.optim_args.metric_name}:.2f}}'
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          mode=mode,
                                          monitor=hparams.optim_args.metric_name,
                                          period=0,
                                          save_top_k=hparams.logger_args.save_top_k)
    
    # TODO: Should be using ddp, but it's running too slowly.
    trainer = Trainer(auto_select_gpus=True,
                      checkpoint_callback=checkpoint_callback,
                      distributed_backend='dp',
                      gpus=hparams.gpu_ids,
                      limit_train_batches=hparams.optim_args.limit_train_batches,
                      logger=logger,
                      max_epochs=hparams.optim_args.num_epochs,
                      val_check_interval=hparams.optim_args.val_check_interval,
                      weights_summary=None)
    wandb.config.update({"phase": "train"})
    trainer.fit(model)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)
