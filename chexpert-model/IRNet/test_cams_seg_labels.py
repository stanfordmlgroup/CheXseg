import sys
import os
sys.path.append(os.path.abspath("../"))

import torch
import numpy as np 
import torch.nn as nn
from torch import multiprocessing, cuda
from misc import torchutils, indexing, imutils
from data.chexpert_dataset_irnet import CheXpertTestCAMDatasetIRNet
from torch.utils.data import DataLoader
from constants import *
import importlib
from tqdm import tqdm
from args.train_arg_parser_irnet import TrainArgParserIRNet
import torch.nn.functional as F
import imageio
from augmentations import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-7

def main(args):
    dataset = CheXpertTestCAMDatasetIRNet()
    data_loader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count(), pin_memory=False)

    with torch.no_grad(), cuda.device(0):
        fg_intersection = np.zeros(len(LOCALIZATION_TASKS))
        fg_union = np.zeros(len(LOCALIZATION_TASKS))

        for iter, pack in enumerate(tqdm(data_loader)):
            img_level_labels = pack['img_level_labels'][0]
            gt_seg_labels = pack['gt_seg_labels'][0]
            cam_seg_labels = pack['cam_seg_labels'][0]
            img_name = pack['base_name'][0]
            
            np.save(os.path.join(CHEXPERT_PARENT_TEST_CAMS_DIR / "gt_seg_labels", f"{img_name}_seg_labels.npy"), gt_seg_labels.cpu().numpy())
            

            for index, task in enumerate(LOCALIZATION_TASKS):
                cam_seg_label = cam_seg_labels[index]
                gt_seg_label = gt_seg_labels[index]

                if img_level_labels[index] == 0:
                    cam_seg_label[:] = 0

                intersection_fg = torch.sum(cam_seg_label * gt_seg_label).numpy()
                union_fg = torch.sum(cam_seg_label).numpy() + torch.sum(gt_seg_label).numpy() - intersection_fg

                fg_intersection[index] += intersection_fg
                fg_union[index] += union_fg
            
    for i in range(len(fg_intersection)):
        fg_iou = (fg_intersection[i] + eps) / (fg_union[i] + eps)
        print(f"Index {i} fg iou {fg_iou}")
    
if __name__ == "__main__":
    parser = TrainArgParserIRNet()
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)

 

    
