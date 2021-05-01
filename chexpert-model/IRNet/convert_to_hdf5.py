import sys
import os
sys.path.append(os.path.abspath("../"))

import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from constants import *

from data.chexpert_dataset_irnet import CheXpertConvertHDF5DatasetIRNet

# Set up the data loader
BATCH_SIZE = 32
dataset = CheXpertConvertHDF5DatasetIRNet()
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count())

images_batch = torch.FloatTensor([])
cams_batch = torch.FloatTensor([])
dset_size = len(data_loader.dataset)

if not os.path.exists(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files')):
    os.makedirs(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files'))
# HDF5 file to create
with h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / ('hdf5_files/train_set'), "w") as f:
    with torch.no_grad():
        # Specifying the shape while creating the dataset
        img_dset = f.create_dataset('cxr_images', shape=(dset_size, 3, 320, 320))
        cams_dset = f.create_dataset('cxr_cams', shape=(dset_size, 10, 1, 320, 320))

        # Encoding required to storing Variable length strings
        dt = h5py.string_dtype(encoding='utf-8')
        base_names_dset = f.create_dataset('cxr_base_names', shape=(dset_size,), dtype=dt)
        index = 0

        for iter, pack in enumerate(tqdm(data_loader)):
            img = pack['img']
            base_name = pack['base_name']
            all_cams = pack['all_cams']

            # for the last iteration, we won't have the full batch 
            if iter > (len(data_loader) - 1):
                start_index = index * BATCH_SIZE
                img_dset[start_index:] = img
                base_names_dset[start_index:] = base_name
                cams_dset[start_index] = all_cams
            else:
                start_index = index * BATCH_SIZE
                end_index = (index + 1) * BATCH_SIZE
                img_dset[start_index:end_index] = img
                base_names_dset[start_index:end_index] = base_name
                cams_dset[start_index:end_index] = all_cams

            index += 1


