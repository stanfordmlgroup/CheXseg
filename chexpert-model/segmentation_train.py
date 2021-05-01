import wandb
from args import SegTrainArgParser
from data import get_seg_loader
import torch
import segmentation_models_pytorch as smp
import util

import numpy as np
import random
import json
from constants import *

def main(hparams):
    np.random.seed(hparams.model_args.seed) # numpy
    torch.manual_seed(hparams.model_args.seed) # cpu
    random.seed(hparams.model_args.seed) # python
    torch.cuda.manual_seed_all(hparams.model_args.seed) # gpu
    torch.backends.cudnn.deterministic = True # cudnn
    torch.backends.cudnn.benchmark = False
    model_fn = util.get_seg_model_fn(hparams.model_args.architecture)
    model = model_fn(encoder_name=hparams.model_args.encoder, 
                    encoder_weights=hparams.model_args.encoder_weights, 
                    encoder_weights_type=hparams.model_args.encoder_weights_type,
                    classes=len(hparams.data_args.classes),
                    activation=None)
    wandb.init(name=hparams.logger_args.wandb_run_name,
               project=hparams.logger_args.wandb_project_name)
    wandb.watch(model, log='all')
    train_loader = get_seg_loader(phase=hparams.data_args.train_set,
                                  data_args=hparams.data_args,
                                  transform_args=hparams.transform_args,
                                  model_args=hparams.model_args,
                                  is_training=True,
                                  semi_supervised=hparams.data_args.semi_supervised)
    valid_loader = get_seg_loader(phase=hparams.data_args.valid_set,
                                  data_args=hparams.data_args,
                                  transform_args=hparams.transform_args,
                                  model_args=hparams.model_args,
                                  is_training=False,
                                  semi_supervised=hparams.data_args.semi_supervised)

    iou_thresholds = np.arange(0, 1, 0.05).tolist()
    loss = smp.utils.losses.ClassAverageDiceLoss(activation=torch.nn.Sigmoid())
    metrics = [
        smp.utils.metrics.IoU(activation=torch.nn.Sigmoid(), thresholds=iou_thresholds),
    ]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=hparams.optim_args.lr),
    ])
    train_epoch = smp.utils.train.TrainEpoch(model, 
                                             loss=loss, 
                                             metrics=metrics, 
                                             optimizer=optimizer,
                                             thresholds=iou_thresholds,
                                             device=hparams.model_args.device,
                                             num_channels=len(hparams.data_args.classes),
                                             verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, 
                                             loss=loss, 
                                             metrics=metrics, 
                                             thresholds=iou_thresholds,
                                             device=hparams.model_args.device,
                                             num_channels=len(hparams.data_args.classes),
                                             verbose=True,)

    max_score = 0
    print(f"Training {len(hparams.data_args.classes)} classes")
    for i in range(hparams.optim_args.num_epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        valid_ious = np.divide(valid_logs['iou_score'][0], valid_logs['iou_score'][1])
        valid_max_ious = np.amax(valid_ious, axis=0)
        if hparams.optim_args.valid_common_pathologies: # use most common pathologies to evaluate and save models
            common_pathologies_ious = np.zeros(len(SEGMENTATION_COMMON_PATHOLOGIES))
            for common_pathology_index, pathology in enumerate(SEGMENTATION_COMMON_PATHOLOGIES):
                max_iou_index = hparams.data_args.classes.index(pathology)
                common_pathologies_ious[common_pathology_index] = valid_max_ious[max_iou_index]
            common_pathologies_miou = np.mean(common_pathologies_ious)
            # to find best thresholding, must use train set since validation set doesn't have most pathologies present
            train_ious = np.divide(train_logs['iou_score'][0], train_logs['iou_score'][1])
            train_max_ious_index = np.argmax(train_ious, axis=0)
            best_thresholds = [iou_thresholds[num] for num in np.nditer(train_max_ious_index)]
        else:
            valid_max_ious_index = np.argmax(valid_ious, axis=0)
            best_thresholds = [iou_thresholds[num] for num in np.nditer(valid_max_ious_index)]
        valid_miou = np.nanmean(valid_max_ious)

        # logging
        logs = {hparams.data_args.classes[i]: valid_max_ious[i] for i in range(len(hparams.data_args.classes))}
        logs.update({"train loss": train_logs['class_average_dice_loss'],
                     "validation loss": valid_logs['class_average_dice_loss'],
                     "validation miou score": valid_miou,})
        wandb.log(logs)

        if hparams.optim_args.valid_common_pathologies:
            if max_score < common_pathologies_miou:
                max_score = common_pathologies_miou
                torch.save(model.state_dict(), hparams.logger_args.save_dir / "best_model.pth")
                with open(hparams.logger_args.save_dir / "thresholds.txt", "w") as threshold_file:
                    json.dump(best_thresholds, threshold_file)
                print(f'Model saved with performance of {common_pathologies_miou} on common pathologies!')
        else:
            if max_score < valid_miou:
                max_score = valid_miou
                torch.save(model.state_dict(), hparams.logger_args.save_dir / "best_model.pth")
                with open(hparams.logger_args.save_dir / "thresholds.txt", "w") as threshold_file:
                    json.dump(best_thresholds, threshold_file)
                print('Model saved!')
        
if __name__ == '__main__':
    parser = SegTrainArgParser()
    hyperparams = parser.parse_args()
    # TRAIN
    main(hyperparams)
