import wandb
from args import SegTestArgParser
import segmentation_models_pytorch as smp
from data import get_seg_loader
import torch
import pandas as pd
import util
import json
from argparse import Namespace
from tqdm import tqdm
from pycocotools import mask
import os

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test(args):
    train_args = args.model_args.config_path + '/args.json'
    ckpt_path = args.model_args.config_path + '/best_model.pth'
    with open(train_args) as f:
        train_args = json.load(f, object_hook=dict_to_namespace)
    model_fn = util.get_seg_model_fn(train_args.model_args.architecture)
    model = model_fn(encoder_name=train_args.model_args.encoder, 
                     encoder_weights=train_args.model_args.encoder_weights, 
                     classes=len(train_args.data_args.classes), 
                     activation=train_args.model_args.activation)
    model.load_state_dict(torch.load(ckpt_path))
    
    thresholds_file = open(args.model_args.config_path + '/thresholds.txt', 'r')
    thresholds = json.load(thresholds_file)

    ## TODO:Customize this with args
    # loss = smp.utils.losses.DiceLoss()
    loss = smp.utils.losses.MyDiceLoss(activation=torch.nn.Sigmoid())
    metrics = [
        smp.utils.metrics.IoUWithThresholds(thresholds, activation=torch.nn.Sigmoid()),
    ]

    classes = train_args.data_args.classes.copy()
    # classes.insert(0, 'Overall')
    # classes_formatted = [x.lower().replace(' ', '_') for x in args.data_args.classes]
    # for i in range(len(args.data_args.classes)):
    #     ignore_channels = list(range(0, len(args.data_args.classes)))
    #     ignore_channels.remove(i)
    #     metrics.append(smp.utils.metrics.IoU(threshold=0.5, ignore_channels=ignore_channels, task=classes_formatted[i]))

    test_loader = get_seg_loader(phase=args.data_args.test_set,
                                 data_args=train_args.data_args,
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
    thresholds_tensor = torch.FloatTensor(thresholds).to(device='cuda')
    predicted_test_seg_maps = {}

    for index, (img, img_id) in tqdm(enumerate(test_loader)):
        img_name = img_id[0]
        img = img.to(device)
        predictions = model.predict(img)
        predicted_seg_maps = smp.utils.functional._taskwise_threshold(predictions, thresholds)

        for i in range(len(classes)):
            task = classes[i]
            predicted_seg_map = np.asfortranarray(predicted_seg_maps[0][i].cpu().numpy().astype('uint8'))
            # print(predicted_seg_map.shape)
            encoded_map = mask.encode(predicted_seg_map)
            encoded_map['counts'] = encoded_map['counts'].decode()


            if img_name in predicted_test_seg_maps:
                if task in predicted_test_seg_maps[img_name]:
                    print(f'Check for duplicates for {task} for {img_name}')
                    break
                else:
                    predicted_test_seg_maps[img_name][task] = encoded_map
            else:
                predicted_test_seg_maps[img_name] = {}
                predicted_test_seg_maps[img_name][task] = encoded_map
        
        with open(os.path.join(args.logger_args.output_labels_save_dir, "pred_semi_supervised_test_seg_labels.json"), "w") as f:
            json.dump(predicted_test_seg_maps, f)
    

        
    # logs = test_epoch.run(test_loader)
    # eps = 1e-7
    # intersection, union = logs['iou_thresh_score']
    # ious = np.divide(intersection, union)
    # miou = np.insert(ious, 0, np.mean(ious))
    # for i in range(len(classes)):
    #     print("Task:", classes[i], "iou:", miou[i])
    # results = pd.DataFrame([miou], columns=classes, index=[args.logger_args.experiment_name])
    # results = results.sort_index(axis=1)
    # results.to_csv(args.logger_args.results_dir / 'results.csv')

    # classes = args.data_args.classes.copy()
    # if args.data_args.task: # only test one task if specified
    #     classes = [args.data_args.task]
    # else:
    #     classes.insert(2, 'Overall')
    # ious = []
    # for task in classes: # test all classes seperately to get individual ious
    #     if task != 'Overall':
    #         test_loader = get_seg_loader(phase=args.data_args.test_set,
    #                                      data_args=args.data_args,
    #                                      transform_args=args.transform_args,
    #                                      model_args=train_args.model_args,
    #                                      is_training=False,
    #                                      task=task)
    #     else:
    #         test_loader = get_seg_loader(phase=args.data_args.test_set,
    #                                      data_args=args.data_args,
    #                                      transform_args=args.transform_args,
    #                                      model_args=train_args.model_args,
    #                                      is_training=False)
    #     test_epoch = smp.utils.train.ValidEpoch(model=model,
    #                                             loss=loss,
    #                                             metrics=metrics,
    #                                             device=args.model_args.device,
    #                                             verbose=False)
    #     logs = test_epoch.run(test_loader)
    #     ious.append(logs['iou_score'])
    #     print('Task:', task, logs)
    # results = pd.DataFrame([ious], columns=classes, index=[args.logger_args.experiment_name])
    # results = results.sort_index(axis=1)
    # results.to_csv(args.logger_args.results_dir / 'results.csv')

def dict_to_namespace(d):
    return Namespace(**d)

if __name__ == "__main__":
    parser = SegTestArgParser()
    test(parser.parse_args())