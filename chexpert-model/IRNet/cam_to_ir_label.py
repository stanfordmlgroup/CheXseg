import sys
import os
sys.path.append(os.path.abspath("../"))

import numpy as np
import imageio
from misc import imutils
from constants import *
import pandas as pd 
import torch
from tqdm import tqdm

def run():
    csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
    cams_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
    ir_labels_directory = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "ir_labels"
    train_set_hdf5_file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_small_set_all"
    
    df = pd.read_csv(Path(csv_path))
    for i in tqdm(range(df.shape[0])):
        base_name = df.iloc[i].values[0]

        img_pkl_file = base_name + "_Cardiomegaly_map.pkl"
        img = pd.read_pickle(cams_path / img_pkl_file)['cxr_img']

        for task in LOCALIZATION_TASKS:
            cam_pkl_file = base_name + f"_{task}_map.pkl"
            cam = pd.read_pickle(cams_path / cam_pkl_file)['map']
            cam = cam.squeeze(0) / torch.max(cam)
            cxr_img = img.permute(1, 2, 0).numpy()
            cxr_img = (cxr_img * 255).round()

            fg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.25)
            fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(cxr_img.astype(np.uint8), fg_conf_cam, n_labels=2)
            keys_cxr = np.array([0, 1])
            fg_conf = keys_cxr[pred]

            bg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.1)
            bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
            pred = imutils.crf_inference_label(cxr_img.astype(np.uint8), bg_conf_cam, n_labels=2)
            bg_conf = keys_cxr[pred]

            ir_label = fg_conf.copy()
            ir_label[fg_conf == 0] = 255
            ir_label[bg_conf + fg_conf == 0] = 0

            imageio.imwrite(os.path.join(ir_labels_directory, base_name +  f"_{task}_ir_label.png"),
                        ir_label.astype(np.uint8))


if __name__ == "__main__":
    run()
