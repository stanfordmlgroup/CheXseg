import pandas as pd
import numpy as np
import json
from pycocotools import mask as pycocomask
import cv2
import random

from torch.utils.data import Dataset as BaseDataset
from constants import *

class SegmentationDataset(BaseDataset):
    def __init__(self, phase, masks_path, 
                 scale, classes, preprocessing=None, include_mask=True):
        self.csv_name = f"{phase}.csv" if not phase.endswith(".csv") else phase
        self.is_train_dataset = self.csv_name == "train.csv"
        self.is_test_dataset = self.csv_name == "test.csv"
        self.is_val_dataset = self.csv_name == "valid.csv"
        self.is_uncertain_dataset = "uncertainty" in self.csv_name
        self.scale = scale
        self.classes = classes
        self.preprocessing = preprocessing
        self.include_mask = include_mask

        if self.is_test_dataset:
            self.csv_path = CHEXPERT_TEST_DIR / f"{phase}_image_paths.csv"
        elif self.is_uncertain_dataset:
            self.csv_path = CHEXPERT_UNCERTAIN_DIR / self.csv_name
        else:
            self.csv_path = CHEXPERT_DATA_DIR / self.csv_name
        self.img_paths = self.get_img_dict()
        
        with open(masks_path) as f:
            self.gt = json.load(f)
        self.img_ids = sorted(self.gt.keys())

    def get_img_dict(self):
        df = pd.read_csv(Path(self.csv_path))[[COL_PATH]]
        df['img_id'] = df[COL_PATH].str[len(CHEXPERT_DATASET_NAME) + len(self.csv_name) - 2:-len('.jpg')].str.replace("/", "_")
        # Prepend the data dir to get the full path.
        df[COL_PATH] = df[COL_PATH].apply(lambda x: CHEXPERT_PARENT_DATA_DIR / x)
        if self.is_test_dataset: #adjust for the fact that images are in codalab
            df[COL_PATH] = df[COL_PATH].apply(lambda p:
                                                Path(str(p).replace(str(CHEXPERT_DATA_DIR),
                                                                    str(CHEXPERT_TEST_DIR))))       
        return pd.Series(df[COL_PATH].values, index=df['img_id']).to_dict()
    
    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img_path = str(self.img_paths[img_id])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize and pad to keep dimension ratio
        old_size = image.shape[:2]
        ratio = float(self.scale)/max(old_size)
        new_size = [int(x*ratio) for x in old_size]
        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = self.scale - new_size[1]
        delta_h = self.scale - new_size[0]
        image = cv2.copyMakeBorder(image, delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2),
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
        
        if self.include_mask:
            masks = [cv2.resize(pycocomask.decode(self.gt[img_id][task]), (new_size[1], new_size[0])) for task in self.classes]
            masks = [cv2.copyMakeBorder(mask, delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2), 
                                        cv2.BORDER_CONSTANT, value=[0,0,0]) for mask in masks]
            mask = np.stack(masks, axis=-1).astype('float')

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            
            return image, mask
        else: # used for distillation when we don't need corresponding label
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            return image, img_id

    def __len__(self):
        return len(self.img_ids)

class SemiSupervisedWeightedDataset(BaseDataset):
    def __init__(self, ss_dataset, ws_dataset, data_args):
        self.data_args = data_args
        self.ss_dataset = ss_dataset
        self.ws_dataset = ws_dataset

    def __getitem__(self, index):
        #print("In semi-supervised weighted")
        if (random.randint(1, 10) * 0.1) <= self.data_args.strong_labels_weight:
            return self.ss_dataset[index % len(self.ss_dataset)]
        else:
            return self.ws_dataset[index % len(self.ws_dataset)]
    
    def __len__(self):
        return len(self.ss_dataset) + len(self.ws_dataset)
