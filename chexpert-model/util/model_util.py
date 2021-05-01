import os
import pandas as pd
import pathlib
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from constants import *

def get_seg_model_fn(architecture):
    if (architecture == 'DeepLabV3Plus'):
        return smp.DeepLabV3Plus
    elif (architecture == 'DeepLabV3'):
        return smp.DeepLabV3
    elif (architecture == 'Unet'):
        return smp.Unet
    elif (architecture == 'UnetPlusPlus'):
        return smp.UnetPlusPlus
    elif (architecture == 'FPN'):
        return smp.FPN
    else:
        raise Exception('Must input valid decoder architecture')

def get_model_folder(ckpt_path):
    model_folder = ckpt_path.split('/')
    model_folder = model_folder[-2] + '_' + model_folder[-1]
    return model_folder


def save_predictions(outputs, probs, model_args, logger_args):
    # get study names
    studies = []
    # saving the train cam paths 
    train_cams_path = []
    for batch in outputs:
        paths = batch['info_dict']['paths']
        for path in paths:
            studies.append(path)
        if logger_args.save_train_cams:
            for train_cam_path in batch['info_dict']['img_paths']:
                train_cams_path += train_cam_path
            
    studies = ['_'.join(item.parts[-POSIX_PATH_PARTS_NUM:]) for item in studies]
    
    # create dataframe
    df = pd.DataFrame(probs.detach().cpu().numpy(), columns=model_args.tasks)
    df['Study'] = studies
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # save predictions.csv
    model_folder = get_model_folder(model_args.ckpt_path)
    directory_path = pathlib.Path(logger_args.save_dir_predictions)/model_folder
    os.makedirs(directory_path, exist_ok=True)
    file_path = directory_path/'predictions.csv'
    df.to_csv(file_path, index=False)

    # If save_train_cams is true, save the path to the train cams in a csv
    if logger_args.save_train_cams:
        # The -4 is for truncating the .jpg at the end
        train_cams_path = ['_'.join(item.parts[-POSIX_PATH_PARTS_NUM_TRAIN_CAMS:])[:-4] for item in train_cams_path]
        train_cams_paths_df = pd.DataFrame(train_cams_path, columns=['Train_CAM_Id'])
        if not os.path.isfile(directory_path/'train_cams_paths.csv'):
            train_cams_paths_df.to_csv(directory_path/'train_cams_paths.csv', index=False)
        else:
            train_cams_paths_df.to_csv(directory_path/'train_cams_paths.csv', mode='a', header=False, index=False)

    
def uncertain_logits_to_probs(logits):
    """Convert explicit uncertainty modeling logits to probabilities P(is_abnormal).

    Args:
        logits: Input of shape (batch_size, num_tasks * 3).

    Returns:
        probs: Output of shape (batch_size, num_tasks).
            Position (i, j) interpreted as P(example i has pathology j).
    """
    b, n_times_d = logits.size()
    d = 3
    if n_times_d % d:
        raise ValueError('Expected logits dimension to be divisible by ' +
                         f'{d}, got size {n_times_d}.')
    n = n_times_d // d

    logits = logits.view(b, n, d)
    probs = F.softmax(logits[:, :, 1:], dim=-1)
    probs = probs[:, :, 1]

    return probs


def get_probs(logits, model_uncertainty):
    if model_uncertainty:
        probs = uncertain_logits_to_probs(logits)
    else:
        probs = torch.sigmoid(logits)
    return probs
