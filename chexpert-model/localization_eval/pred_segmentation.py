# Converts prediction to segmentation for each method.
# Stored as encoded COCO binary mask in a json file. 

# sample run:
# python3 pred_segmentation.py --phase valid --method gradcam --model_type single

from eval_helper import cam_to_segmentation, segmentation_to_mask
from eval_constants import *

import json
import pickle
import numpy as np
from pathlib import Path
from pycocotools import mask
import torch.nn.functional as F
import torch

from argparse import ArgumentParser


def pkl_to_mask(pkl_path, task, phase, smoothing = False, force_negative = False):
    """
    # load cam pickle file, get saliency map and resize. 
    Convert to binary segmentation mask and output encoded mask
    
    Args:
        pkl_path(str): path to the pickle file
        task(str): pathology
        smoothing(bool): if we use smoothing on the heatmap (for IG only)
        force_negative(bool): if we use manually chosen thresholding
    """
    # load pickle file, get saliency map and resize
    info = pickle.load(open(pkl_path,'rb'))
    saliency_map = info['map']
    img_dims = info['cxr_dims']
    
    if phase == 'train':
        map_resized = saliency_map
    else:
        map_resized = F.interpolate(saliency_map, size=(img_dims[1],img_dims[0]), mode='bilinear', align_corners=False)
    
    # convert to segmentation
    try:
        if force_negative:
            override_negative = saliency_map.max() < GRADCAM_CUTOFF[task]
            segm_map = cam_to_segmentation(map_resized, smoothing = smoothing,override_negative = override_negative)
        else:
            segm_map = cam_to_segmentation(map_resized, smoothing)
    except:
        print(f'Error at {img_id}, index = {idx}')
        raise
        
    segm_map = np.array(segm_map,dtype = "int")
    encoded_map = segmentation_to_mask(segm_map)
    return encoded_map


def map_to_results(map_dir, result_name, phase, smoothing = False, force_negative = False):
    """
    Converts saliency maps to result json format. 
    
    Args:
        map_dir(str): path to the directory that stores CAMs pickle files
        result_name(str): name of the prediction file
        smoothing(bool): if we use smoothing
        force_negative(bool): if we use manual thresholding
    """

    print("Parsing saliency maps to evaluation format")
    all_paths = list(Path(map_dir).rglob("*_map.pkl"))
    results = {}

    for idx,pkl_path in enumerate(all_paths):
        
        if idx % 100 ==0:
            print(f'Converting the {idx}th image')
        
        # break down path to image name and task
        path = str(pkl_path).split('/')
        task = path[-1].split('_')[-2]
        img_id = '_'.join(path[-1].split('_')[:-2])
        if task not in LOCALIZATION_TASKS:
            continue
        
        # get encoded segmentation mask
        encoded_map = pkl_to_mask(pkl_path, task, phase, smoothing, force_negative = force_negative)
        
        # add image and segmentation to submission dictionary
        if img_id in results:
            if task in results[img_id]:
                print(f'Check for duplicates for {task} for {img_id}')
                break
            else:
                results[img_id][task] = encoded_map
        else:
            results[img_id] = {}
            results[img_id][task] = encoded_map
    
    # save to json
    print(f'Processed {len(results)} images')
    with open(result_name, "w") as f:
        json.dump(results, f)
    print(f'Ready for evaluation at {result_name}')
    
        

        
if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--phase', type=str, required=True,
                        help='valid or test or train')
    parser.add_argument('--map_dir', type=str, 
                        help='path to the CAMs, only used when phase is train', default=None)
    parser.add_argument('--method', type=str, required=True,
                        help='localization method: gradcam')
    parser.add_argument('--model_type', default='ensemble',
                        help='single or ensemble')
    parser.add_argument('--if_threshold', default=False,
                        help='if using thresholding')
    
    args = parser.parse_args()
    
    method = args.method
    model_type = args.model_type
    map_dir = args.map_dir
    phase = args.phase
    if_threshold = args.if_threshold

    if phase == 'train':
        if if_threshold:
            result_name = f'{map_dir}/seg_labels_encoded_treshold.json'
        else:
            result_name = f'{map_dir}/seg_labels_encoded.json'
    else:
        # get directory that stores the saliency maps
        cam_dirs = valid_cam_dirs if phase == 'valid' else test_cam_dirs
        
        map_dir = cam_dirs[f'{method}_{model_type}']
        
        # create dir 
        result_dir = '/deep/group/aihc-bootcamp-spring2020/localize/eval_results'
        Path(f'{result_dir}/{method}').mkdir(parents=True, exist_ok = True)
        
        if if_threshold:
            result_name = f'{result_dir}/{method}/{phase}_{method}_{model_type}_encoded_threshold.json'
        else:
            result_name = f'{result_dir}/{method}/{phase}_{method}_{model_type}_encoded.json'
    
    if_smoothing = 'ig' in method
    map_to_results(map_dir, result_name, phase, smoothing = if_smoothing, force_negative = if_threshold)