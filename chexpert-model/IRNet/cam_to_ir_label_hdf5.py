import sys
import os
sys.path.append(os.path.abspath("../"))

import numpy as np
import imageio
from misc import imutils
from constants import *
import pandas as pd 
import torch
import h5py
from tqdm import tqdm

def run():
    # csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
    # cams_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
    train_set_hdf5_file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_set"
    ir_labels_hdf5_file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/ir_labels"

    f = h5py.File(train_set_hdf5_file)
    cams_dset = f['cxr_cams']
    images_dset = f['cxr_images']
    # ir_labels_file  = h5py.File(ir_labels_hdf5_file, "w")

    index = 0
    dset_size = cams_dset.shape[0]
    # ir_labels_dset = ir_labels_file.create_dataset("ir_labels", shape=(ten_percent_dset_size, 10, 320, 320))

    with h5py.File(ir_labels_hdf5_file, "w") as ir_labels_file:
        ir_labels_dset = ir_labels_file.create_dataset("ir_labels", shape=(dset_size, 10, 320, 320), dtype='i')
        for i in tqdm(range(dset_size)):
            img = images_dset[i]
            ir_labels_temp = np.array([])
            for j, _ in enumerate(LOCALIZATION_TASKS):
                cam = cams_dset[i][j]
                cam = cam/ np.max(cam)
                cxr_img = img.transpose([1, 2, 0])
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
                # print(ir_label)
                ir_label = np.expand_dims(ir_label, 0)

                if len(ir_labels_temp) == 0:
                    ir_labels_temp = ir_label
                else:
                    ir_labels_temp = np.concatenate((ir_labels_temp, ir_label), axis=0) 
                
            ir_labels_dset[i] = ir_labels_temp

    f.close()

if __name__ == "__main__":
    run()
