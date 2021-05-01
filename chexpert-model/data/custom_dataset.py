import cv2
import torch
import pandas as pd
from PIL import Image

import util
from .base_dataset import BaseDataset
from constants import *


class CustomDataset(BaseDataset):
    def __init__(self, csv_name, is_training, study_level,
                 transform_args, toy, return_info_dict, logger=None, data_args=None, stability_training=False):
        # Pass in parent of data_dir because test set is in a different
        # directory due to dataset release, and uncertain maps are in a
        # different directory as well (both are under the parent directory).
        super().__init__(csv_name, is_training, transform_args)
        self.study_level = study_level
        self.toy = toy
        self.return_info_dict = return_info_dict
        self.logger = logger
        self.data_args = data_args
        self.stability_training = stability_training

        self.is_train_dataset = self.csv_name == "train.csv"
        self.is_test_dataset = self.csv_name == "test.csv"
        self.is_val_dataset = self.csv_name == "valid.csv"
        self.is_uncertain_dataset = "uncertain" in self.csv_name

        if self.is_train_dataset:
            self.csv_path = self.data_args.csv
        elif self.is_uncertain_dataset:
            self.csv_path = self.data_args.uncertain_map_path
        elif self.is_val_dataset:
            self.csv_path = self.data_args.csv_dev
        elif self.is_test_dataset: # Custom test
            if self.data_args.together: # one csv for all of test
                self.csv_path = self.data_args.test_csv
            else:
                self.csv_path = self.data_args.paths_csv

        if self.is_val_dataset:
            print("valid", self.csv_path)

        df = self.load_df()

        self.studies = df[COL_STUDY].drop_duplicates()

        if self.toy and self.csv_name == 'train.csv':
            self.studies = self.studies.sample(n=10)
            df = df[df[COL_STUDY].isin(self.studies)]
            df = df.reset_index(drop=True)

        # Set Study folder as index.
        if self.study_level:
            self.set_study_as_index(df)

        self.labels = self.get_labels(df)
        self.img_paths = self.get_paths(df)

    def load_df(self):
        df = pd.read_csv(Path(self.csv_path))
        df[COL_STUDY] = df[COL_PATH].apply(lambda p: Path(p).parent)
        if self.is_test_dataset and not self.data_args.together:
            if self.data_args.custom:
                gt_df = pd.read_csv(self.data_args.gt_csv)
            df = pd.merge(df, gt_df, on=COL_STUDY, how = 'outer')
            df = df.dropna(subset=['Path'])

        df = df.rename(columns={"Lung Opacity": "Airspace Opacity"}).sort_values(COL_STUDY)

        df[CHEXPERT_TASKS] = df[CHEXPERT_TASKS].fillna(value=0)

        return df

    def set_study_as_index(self, df):
        df.index = df[COL_STUDY]

    def get_paths(self, df):
        return df[COL_PATH]

    def get_labels(self, df):
        # Get the labels
        if self.study_level:
            study_df = df.drop_duplicates(subset=COL_STUDY)
            labels = study_df[CHEXPERT_TASKS]
        else:
            labels = df[CHEXPERT_TASKS]

        return labels

    def get_study(self, index):

        # Get study folder path
        study_path = self.studies.iloc[index]

        # Get and transform the label
        label = self.labels.loc[study_path].values
        label = torch.FloatTensor(label)

        # Get and transform the images
        # corresponding to the study at hand
        img_paths = pd.Series(self.img_paths.loc[study_path]).tolist()
        imgs = []
        #print(img_paths)
        from PIL import ExifTags
        import numpy as np
        for path in img_paths:
            img = util.rotate_img(path)
            if self.data_args.channels:
                from PIL import ImageEnhance
                #brightness up, contrast up, sharpen
                img = img.convert("L")
                up_channel = ImageEnhance.Brightness(img).enhance(1.1)
                up_channel = ImageEnhance.Sharpness(up_channel).enhance(1.2)
                up_channel = ImageEnhance.Contrast(up_channel).enhance(1.1).convert("L")
                down_channel = ImageEnhance.Brightness(img).enhance(0.9)
                down_channel = ImageEnhance.Sharpness(down_channel).enhance(1.0)
                down_channel = ImageEnhance.Contrast(down_channel).enhance(0.9).convert("L")
                img = Image.merge("RGB",(img,up_channel,down_channel))
            imgs.append(img)

        img_dims = [img.size for img in imgs]
        imgs = [self.transform(img) for img in imgs]
        imgs = torch.stack(imgs)

        if self.return_info_dict:
            info_dict = {'paths': study_path,
                         'img_paths': img_paths,
                         'img_dims': img_dims}
            return imgs, label, info_dict

        return imgs, label

    def get_image(self, index):

        # Get and transform the label
        label = self.labels.iloc[index].values
        label = torch.FloatTensor(label)

        # Get and transform the image
        img_path = self.img_paths.iloc[index]

        from PIL import ExifTags
        img = util.rotate_img(img_path)
        if self.data_args.channels:
            from PIL import ImageEnhance
            #brightness up, contrast up, sharpen
            img = img.convert("L")
            up_channel = ImageEnhance.Brightness(img).enhance(1.1)
            up_channel = ImageEnhance.Sharpness(up_channel).enhance(1.2)
            up_channel = ImageEnhance.Contrast(up_channel).enhance(1.1).convert("L")
            down_channel = ImageEnhance.Brightness(img).enhance(0.9)
            down_channel = ImageEnhance.Sharpness(down_channel).enhance(1.0)
            down_channel = ImageEnhance.Contrast(down_channel).enhance(0.9).convert("L")
            img = Image.merge("RGB",(img,up_channel,down_channel))

        img_dims = img.size
        img = self.transform(img)

        if self.return_info_dict:
            info_dict = {'paths': str(img_path),
                         'img_dims': img_dims}
            return img, label, info_dict

        return img, label

    def __getitem__(self, index):
        if self.study_level:
            return self.get_study(index)
        else:
            return self.get_image(index)

