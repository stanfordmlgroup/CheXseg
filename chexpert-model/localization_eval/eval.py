import numpy as np
from PIL import Image
from pycocotools import mask
import pandas as pd

import json
from pathlib import Path
from argparse import ArgumentParser
from eval_helper import create_mask

LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Support Devices"
                  ]

def iou_seg(mask1,mask2):
    """
    Calculate iou scores of two segmentation masks
    
    Args: 
        mask1 (np.array): binary segmentation mask
        mask2 (np.array): binary segmentation mask
    Returns:
        iou score (a scalar)
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    if np.sum(union) ==0:
        iou_score = -1
#     elif np.sum(mask2) ==0:
#         iou_score = -2
    else:
        iou_score = np.sum(intersection) / (np.sum(union))
    return iou_score


def compute_metrics(gt_dir, pred_dir):
    """
    Take in ground truth and prediction json (both encoded) and return iou for each image under each pathology
    """
    with open(gt_dir) as f:
        gt = json.load(f)
        
    with open(pred_dir) as f:
        pred = json.load(f)
        
    ious = {}

    all_ids = sorted(pred.keys())
    tasks = sorted(LOCALIZATION_TASKS)

    for task in tasks:
        print(f'Evaluating {task}')
        ious[task] = []
   
        for img_id in all_ids:
             
            # get predicted segmentation mask
            pred_item = pred[img_id][task]
            pred_mask = mask.decode(pred_item)
            
            # get ground_truth segmentation mask
            # if image not in gt, create zero mask
            if img_id not in gt:
                gt_mask = np.zeros(pred_item['size'])
            else:
                gt_item = gt[img_id][task]
    #             gt_item['counts'] = gt_item['counts'].encode()
                gt_mask = mask.decode(gt_item)
            
            assert gt_mask.shape == pred_mask.shape
            
            iou_score = iou_seg(pred_mask, gt_mask)
            ious[task].append(iou_score)
        
        if np.all(ious[task]==-1):
            print(f'{task} has all true negatives')
        assert len(ious[task]) == len(pred.keys())
            
    return ious


def bootstrap_metric(df, num_replicates, metric = 'iou'):
    """
    Create dataframe of bootstrap samples 
    """
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df.columns:
            if metric == 'iou':
                performance = df_replicate[df_replicate>-1][task].mean()
            else:
                performance = df_replicate.mean()
            replicate_performances[task] = performance
        return replicate_performances
    
    all_performances = []
    
    for _ in range(num_replicates):
        replicate_performances = single_replicate_performances()
        all_performances.append(replicate_performances)

    df_performances = pd.DataFrame.from_records(all_performances)
    return df_performances  


def compute_cis(series, confidence_level):
    """
    Compute confidence intervals given cf level
    """
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(),3)
    return lower, mean, upper

def create_ci_record(perfs, name):
    lower, mean, upper = compute_cis(perfs, confidence_level = 0.05)
    record = {"name": name,
              "lower": lower,
              "mean": mean,
              "upper": upper,
                  }
    return record     


def evaluate(gt_dir, pred_dir,save_dir,table_name):
    """
    Pipeline to evaluate localizations. Return miou by pathologies and their confidence intervals
    """
    
    # create save dir if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    
    metrics = compute_metrics(gt_dir, pred_dir)
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.to_csv(f'{save_dir}/{table_name}_merged_iou.csv',index = False)
    
    bs_df = bootstrap_metric(metrics_df,1000)
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))
   
    summary_df = pd.DataFrame.from_records(records)
    summary_df.to_csv(f'{save_dir}/{table_name}_merged_summary.csv',index = False)


if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument('--method', type=str, required=True,
                        help='gradcam, ig, wildcat, human')
    parser.add_argument('--phase', type=str, required=True,
                        help='enter valid or test')
    parser.add_argument('--model_type', type=str, default = 'ensemble',
                        help='single, ensemble or baseline')
    
    args = parser.parse_args()
    
    method = args.method
    phase = args.phase
    model_type = args.model_type
    
    group_dir = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_path = f'{group_dir}/annotations/{phase}_encoded.json'
    
    if method == 'human':
        assert phase == 'test'
        pred_path = f'{group_dir}/annotations/vietnam_encoded.json'
        table_name = f'{phase}_{method}'
    else:
        pred_path = f'{group_dir}/eval_results/{method}/{phase}_{method}_{model_type}_encoded.json'
        table_name = f'{phase}_{method}_{model_type}'
        
        if method == "ig":
            pred_path = f'{group_dir}/eval_results/{method}/{phase}_{method}_{model_type}_encoded_largek.json'
            
    
    save_dir = f'{group_dir}/eval_results/{method}'
    evaluate(gt_path, pred_path, save_dir,table_name)
