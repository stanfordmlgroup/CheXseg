import sys
import os
sys.path.append(os.path.abspath("../"))

from constants import *
import numpy as np
import h5py

TRAIN_CAM_DIR = Path("/scr/objectefficiency/train_cams_ten_percent_imagenet_norm/2ywovex5_epoch=2-chexpert_competition_AUROC=0.89.ckpt")
ten_percent_dset_size = 22341

with h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set_ten_percent_combined'), "w") as f:
    images_dataset = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set_ten_percent' + "_" + str(0)), 'r').get("cxr_images_ten_percent")[:, :]
    cams_dataset = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set_ten_percent' + "_" + str(0)), 'r').get("cxr_cams_ten_percent")[:, :]
   
    dset_cams = f.create_dataset("cams", data=cams_dataset, maxshape=(None, None, None, None, None))
    dset_images = f.create_dataset("images", data=images_dataset, maxshape=(None, None, None, None))

    dset_cams.resize(ten_percent_dset_size, axis=0)
    dset_images.resize(ten_percent_dset_size, axis=0)
    index = 0
    for i in range(100, 699 + 100, 100):
        print(i)
        if i > 698:
            out_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set_ten_percent' + "_" + str(698))
            with h5py.File(out_path, "r") as f:
                start_index = 32 + (3200 * index)
                dset_cams[start_index:] = f.get("cxr_cams_ten_percent")[:,:]
                dset_images[start_index:] = f.get("cxr_images_ten_percent")[:,:]
        else:
            out_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set_ten_percent' + "_" + str(i))
            with h5py.File(out_path, "r") as f:
                start_index = 32 + (3200 * index)
                end_index = 32 + (3200 * (index + 1))
                dset_cams[start_index:end_index] = f.get("cxr_cams_ten_percent")[:,:]
                dset_images[start_index:end_index] = f.get("cxr_images_ten_percent")[:,:]
        index += 1