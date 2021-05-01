import sys
import os
sys.path.append(os.path.abspath("../"))

import torch
import numpy as np 
import torch.nn as nn
from torch import multiprocessing, cuda
from misc import torchutils, indexing, imutils
from data.chexpert_dataset_irnet import CheXpertMSDatasetIRNet, CheXpertMSDatasetHDF5IRNet
from torch.utils.data import DataLoader
from constants import *
import importlib
from tqdm import tqdm
from args.train_arg_parser_irnet import TrainArgParserIRNet
import torch.nn.functional as F
import imageio
from augmentations import get_transforms
from pycocotools import mask
import json

# seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_config = {
        'size': 320,
        'augmentation_scope': 'none',
        'images_normalization': 'default',
        'images_output_format_type': 'float',
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
    eps = 1e-7

    model = getattr(importlib.import_module(irnet_args.network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(IRNET_MODEL_SAVE_DIR / irnet_args.best_model_name), strict=False)
    model.eval()

    dataset = CheXpertMSDatasetHDF5IRNet("train.csv", irnet_args.is_training, None, transform)
    data_loader = DataLoader(dataset, shuffle=False, num_workers=os.cpu_count(), pin_memory=False)

    with torch.no_grad(), cuda.device(0):
        #model.cuda()
        model = model.to(device)
        all_intersection = np.zeros(len(LOCALIZATION_TASKS))
        all_union = np.zeros(len(LOCALIZATION_TASKS))
        fg_intersection = np.zeros(len(LOCALIZATION_TASKS))
        fg_union = np.zeros(len(LOCALIZATION_TASKS))
        results_irnet = {}
        results_cam = {}


        if not os.path.exists(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "gt_seg_labels"):
            os.makedirs(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "gt_seg_labels")

        for iter, pack in enumerate(tqdm(data_loader)):
            img = pack['img'][0]
            img_name = pack['base_name'][0]

            orig_img_size = np.asarray(pack['size'])
            # print("ORIG IMG SIZE=", orig_img_size)
            img_level_labels = pack['img_level_labels'][0].cpu().numpy()
            all_cams = pack['all_cams']
            # print(pack['img'][0].shape)
            #x = torch.cat([pack['img'][0], pack['img'][0].flip(-1)], dim=0)
            edge, dp = model(img.to(device))
            
            gt_seg_labels = pack['gt_seg_labels'][0]
            if irnet_args.is_training:
                np.save(os.path.join(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "gt_seg_labels", f"{img_name}_seg_labels.npy"), gt_seg_labels.cpu().numpy())
            else:
                np.save(os.path.join(CHEXPERT_PARENT_TEST_CAMS_DIR / "gt_seg_labels", f"{img_name}_seg_labels.npy"), gt_seg_labels.cpu().numpy())
            # print("GT SEG LABELS SHAPE=", gt_seg_labels.shape)
            
            for index, task in enumerate(LOCALIZATION_TASKS):
                cam_downsized_value = all_cams[:, index, :, :].cuda()
                #print("CAM Downsized Values shape", cam_downsized_value.shape)
                rw = indexing.propagate_to_edge(cam_downsized_value, edge, beta=5, exp_times=3,
                                             radius=3)
            
                rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[1],
                    :orig_img_size[0]]

                rw_up = rw_up / torch.max(rw_up)
                #print(rw_up[0].cpu().numpy().astype(np.uint8))
                rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=0.6)
                # print(rw_up_bg[1])
                # print(rw_up_bg)
                rw_pred = torch.argmax(rw_up_bg, dim=0)

                pseudo_seg_label_irnet = np.asfortranarray(rw_pred.cpu().numpy().astype('uint8'))
                encoded_map = mask.encode(pseudo_seg_label_irnet)
                encoded_map['counts'] = encoded_map['counts'].decode()

                # add image and segmentation to submission dictionary
                if img_name in results_irnet:
                    if task in results_irnet[img_name]:
                        print(f'Check for duplicates for {task} for {img_name}')
                        break
                    else:
                        results_irnet[img_name][task] = encoded_map
                else:
                    results_irnet[img_name] = {}
                    results_irnet[img_name][task] = encoded_map
                
                pseudo_seg_label_cams = np.asfortranarray(gt_seg_labels[index].cpu().numpy().astype('uint8'))
                encoded_map_cams = mask.encode(pseudo_seg_label_cams)
                encoded_map_cams['counts'] = encoded_map_cams['counts'].decode()

                # add image and segmentation to submission dictionary
                if img_name in results_cam:
                    if task in results_cam[img_name]:
                        print(f'Check for duplicates for {task} for {img_name}')
                        break
                    else:
                        results_cam[img_name][task] = encoded_map_cams
                else:
                    results_cam[img_name] = {}
                    results_cam[img_name][task] = encoded_map_cams
                
                #pseudo_labels_save_dir_name = os.path.join(irnet_args.pseudo_labels_save_dir, "pseudo_seg_labels_irnet")
                #imageio.imsave(os.path.join(pseudo_labels_save_dir_name, f"{img_name}_{task}_pseudo_seg_labels.png"), rw_pred.cpu().numpy().astype(np.uint8))
            
            # print(fg_intersection)
            # print(fg_union)

        with open(os.path.join(irnet_args.pseudo_labels_save_dir, "pseudo_seg_labels_encoded_irnet.json"), "w") as f:
            json.dump(results_irnet, f)

        with open(os.path.join(irnet_args.pseudo_labels_save_dir, "pseudo_seg_labels_encoded_cams.json"), "w") as f:
            json.dump(results_cam, f)

    
    print("50 3 3")
    for i in range(len(all_intersection)):
        # all_iou = (all_intersection[i] + eps) / (all_union[i] + eps)
        # print(f"Index {i} all iou {all_iou}")
        fg_iou = (fg_intersection[i] + eps) / (fg_union[i] + eps)
        print(f"Index {i} fg iou {fg_iou}")

if __name__ == "__main__":
    parser = TrainArgParserIRNet()
    hyperparams = parser.parse_args()

    # TRAIN
    main(hyperparams)
