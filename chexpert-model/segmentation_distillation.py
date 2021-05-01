import wandb
from args import SegDistArgParser
import segmentation_models_pytorch as smp
from data import get_dist_loader
from data import get_seg_loader
import torch
import pandas as pd
import util
import json
from argparse import Namespace

import numpy as np
import random
from constants import *

def main(args):
    np.random.seed(args.model_args.seed) # numpy
    torch.manual_seed(args.model_args.seed) # cpu
    random.seed(args.model_args.seed) # python
    torch.cuda.manual_seed_all(args.model_args.seed) # gpu
    torch.backends.cudnn.deterministic = True # cudnn
    config_path = args.model_args.config_path
    teacher_model_args = config_path + '/args.json'
    teacher_ckpt_path = config_path + '/best_model.pth'
    with open(teacher_model_args) as f:
        teacher_train_args = json.load(f, object_hook=dict_to_namespace)
    if not teacher_train_args.model_args.encoder_weights:
        print("Initializing teacher and student model with random weights.")
    teacher_model_fn = util.get_seg_model_fn(teacher_train_args.model_args.architecture)
    teacher_model = teacher_model_fn(encoder_name=teacher_train_args.model_args.encoder, 
                                     encoder_weights=teacher_train_args.model_args.encoder_weights, 
                                     encoder_weights_type=teacher_train_args.model_args.encoder_weights_type,
                                     classes=len(teacher_train_args.data_args.classes), 
                                     activation=None)
    teacher_model.load_state_dict(torch.load(teacher_ckpt_path))

    student_model_fn = util.get_seg_model_fn(teacher_train_args.model_args.architecture)
    student_model = student_model_fn(encoder_name=teacher_train_args.model_args.encoder, 
                                     encoder_weights=teacher_train_args.model_args.encoder_weights, 
                                     encoder_weights_type=teacher_train_args.model_args.encoder_weights_type,
                                     classes=len(teacher_train_args.data_args.classes), 
                                     activation=None)

    wandb.init(name=args.logger_args.wandb_run_name,
               project=args.logger_args.wandb_project_name)
    wandb.watch(student_model, log='all')
    loss = smp.utils.losses.DistillationLoss(temp=args.optim_args.temperature)
    iou_thresholds = np.arange(0, 1, 0.05).tolist()
    metrics = [smp.utils.metrics.IoU(activation=torch.nn.Sigmoid(), thresholds=iou_thresholds)]
    optimizer = torch.optim.Adam([ 
        dict(params=student_model.parameters(), lr=args.optim_args.lr),
    ])
    train_loader = get_dist_loader(model=teacher_model, masks_path=args.data_args.masks_path, 
                                   encoder=teacher_train_args.model_args.encoder,
                                   encoder_weights=teacher_train_args.model_args.encoder_weights,
                                   data_args=teacher_train_args.data_args,
                                   transform_args=args.transform_args,
                                   model_args=teacher_train_args.model_args,
                                   is_training=True,
                                   images_path=args.data_args.images_path,)
    valid_loader = get_seg_loader(phase=teacher_train_args.data_args.valid_set,
                                  data_args=teacher_train_args.data_args,
                                  transform_args=teacher_train_args.transform_args,
                                  model_args=teacher_train_args.model_args,
                                  is_training=False,)

    train_epoch = smp.utils.train.TrainEpoch(student_model, 
                                             loss=loss, 
                                             metrics=metrics, 
                                             optimizer=optimizer,
                                             thresholds=iou_thresholds,
                                             device=teacher_train_args.model_args.device,
                                             num_channels=len(teacher_train_args.data_args.classes),
                                             verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(student_model, 
                                             loss=loss, 
                                             metrics=metrics, 
                                             thresholds=iou_thresholds,
                                             device=teacher_train_args.model_args.device,
                                             num_channels=len(teacher_train_args.data_args.classes),
                                             verbose=True)

    max_score = 0
    if teacher_train_args.optim_args.valid_common_pathologies:
        valid_common_pathologies = SEGMENTATION_COMMON_PATHOLOGIES
    else:
        valid_common_pathologies = None
    print(f"Training {len(teacher_train_args.data_args.classes)} classes")
    for i in range(0, args.optim_args.num_epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader, 
                                     valid_epoch=valid_epoch, 
                                     valid_loader=valid_loader, 
                                     num_valid_per_epoch=args.logger_args.num_valid_per_epoch, 
                                     max_score=max_score, 
                                     classes=teacher_train_args.data_args.classes, 
                                     save_dir=args.logger_args.save_dir,
                                     valid_common_pathologies=valid_common_pathologies)
        max_score = train_logs['max_score']

def dict_to_namespace(d):
    return Namespace(**d)

if __name__ == '__main__':
    parser = SegDistArgParser()
    main(parser.parse_args())