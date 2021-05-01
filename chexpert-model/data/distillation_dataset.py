import pandas as pd
import numpy as np
import json
from pycocotools import mask as pycocomask
import cv2
from tqdm import tqdm
import os
from pathlib import Path

from torch.utils.data import Dataset as BaseDataset
from constants import *

import torch
import h5py
import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from .segmentation_dataset import SegmentationDataset


class DistillationDataset(BaseDataset):
    def __init__(self, model, masks_path, encoder, encoder_weights, scale, classes, images_path=None, preprocessing=None):
        self.model = model
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.csv_name = "train.csv"
        self.csv_path = CHEXPERT_DATA_DIR / self.csv_name
        self.preprocessing = preprocessing
        self.scale = scale
        self.classes = classes

        if images_path:
            self.create_masks_json(masks_path, images_path)
        
        if not masks_path:
            raise Error('Must include a mask path')
        hf = h5py.File(masks_path, 'r')
        self.img_ids = hf.get('crx_base_names')
        self.imgs = hf.get('crx_images')
        self.masks = hf.get('model_preds')
    
    def __getitem__(self, i):
        return self.imgs[i], self.masks[i]

    def __len__(self):
        return len(self.img_ids)

    def create_masks_json(self, out, images_path):
        print("Saving output predictions of teacher model...")
        with open(images_path) as temp:
            gt = json.load(temp)

        batch_size = 8
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        images_dataset = SegmentationDataset('train', images_path, self.scale, 
                                             self.classes, preprocessing=get_preprocessing(preprocessing_fn), include_mask=False)
        images_loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        dset_size = len(images_loader.dataset)

        if not os.path.exists(Path(out).parent.absolute()):
            os.makedirs(Path(out).parent.absolute())
        with h5py.File(out, 'w') as f:
            with torch.no_grad():
                img_dset = f.create_dataset('crx_images', shape=(dset_size, 3, 320, 320))
                model_preds_dset = f.create_dataset('model_preds', shape=(dset_size, 10, 320, 320))
                base_names_dset = f.create_dataset('crx_base_names', shape=(dset_size,), dtype=h5py.string_dtype(encoding='utf-8'))
                for iter, pack in enumerate(tqdm(images_loader)):
                    img = pack[0]
                    pr_mask = self.model.predict(img)
                    base_name = pack[1]
                    
                    start_index = iter * batch_size
                    if iter == (len(images_loader) - 1):
                        img_dset[start_index:] = img
                        model_preds_dset[start_index:] = pr_mask
                        base_names_dset[start_index:] = base_name
                    else:
                        img_dset[start_index:start_index+batch_size] = img
                        model_preds_dset[start_index:start_index+batch_size] = pr_mask
                        base_names_dset[start_index:start_index+batch_size] = base_name

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

    
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)