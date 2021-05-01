import wandb
from args import SegTestArgParser
import segmentation_models_pytorch as smp
from data import get_seg_loader
import torch
import pandas as pd
import util
import json
from argparse import Namespace

import numpy as np

def test(args):
    if args.model_args.distillation_teacher_config_path:
        train_args = args.model_args.distillation_teacher_config_path + '/args.json'
        ckpt_path = args.model_args.config_path + '/distilled_model.pth'
    else:
        train_args = args.model_args.config_path + '/args.json'
        ckpt_path = args.model_args.config_path + '/best_model.pth'
    with open(train_args) as f:
        train_args = json.load(f, object_hook=dict_to_namespace)
    model_fn = util.get_seg_model_fn(train_args.model_args.architecture)
    model = model_fn(encoder_name=train_args.model_args.encoder, 
                     encoder_weights=train_args.model_args.encoder_weights, 
                     encoder_weights_type=train_args.model_args.encoder_weights_type,
                     classes=len(train_args.data_args.classes), 
                     activation=None)
    model.load_state_dict(torch.load(ckpt_path))
    
    if args.model_args.distillation_teacher_config_path:
        thresholds_file = open(args.model_args.config_path + '/distilled_thresholds.txt', 'r')
    else:
        thresholds_file = open(args.model_args.config_path + '/thresholds.txt', 'r')
    thresholds = json.load(thresholds_file)

    loss = smp.utils.losses.ClassAverageDiceLoss(activation=torch.nn.Sigmoid())
    metrics = [
        smp.utils.metrics.IoUWithThresholds(thresholds, activation=torch.nn.Sigmoid()),
    ]

    classes = train_args.data_args.classes.copy()
    classes.insert(0, 'Overall')
    args.data_args.classes = train_args.data_args.classes
    
    test_loader = get_seg_loader(phase=args.data_args.test_set,
                                 data_args=args.data_args,
                                 transform_args=args.transform_args,
                                 model_args=train_args.model_args,
                                 is_training=False)
    test_epoch = smp.utils.train.ValidEpoch(model=model,
                                            loss=loss,
                                            metrics=metrics,
                                            thresholds=thresholds,
                                            device=args.model_args.device,
                                            num_channels=len(train_args.data_args.classes),
                                            verbose=True)
    logs = test_epoch.run(test_loader)
    intersection, union = logs['iou_thresh_score']
    ious = np.divide(intersection, union)
    miou = np.insert(ious, 0, np.mean(ious))
    for i in range(len(classes)):
        print("Task:", classes[i], "iou:", miou[i])
    results = pd.DataFrame([miou], columns=classes, index=[args.logger_args.experiment_name])
    results = results.sort_index(axis=1)
    results.to_csv(args.logger_args.results_dir / 'results.csv')

def dict_to_namespace(d):
    return Namespace(**d)

if __name__ == "__main__":
    parser = SegTestArgParser()
    test(parser.parse_args())