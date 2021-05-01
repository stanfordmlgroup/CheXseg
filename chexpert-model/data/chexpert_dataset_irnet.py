import os
import torch
import pandas as pd 
from .base_dataset import BaseDataset
from constants import *
from pycocotools import mask
import json
import numpy as np 
from IRNet.misc import imutils
from localization_eval.eval_constants import *
import torch.nn.functional as F
import torchvision.transforms as t 
import imageio
from PIL import Image
from torch.utils.data import Dataset
import h5py


class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):
        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):
        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label) 

class CheXpertDatasetIRNet(BaseDataset):
    def __init__(self, csv_name, is_training,
                 transform_args, transform, indices_from, indices_to, logger=None, data_args=None):
        super().__init__(csv_name, is_training, transform_args)
        self.csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
        self.img_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
        self.seg_labels_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "seg_labels_encoded.json"
        self.ir_labels_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "ir_labels"
        self.df = pd.read_csv(Path(self.csv_path))
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)
        self.transform = transform
        self.hdf5file_train_set = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_set")
        self.hdf5file_ir_labels = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/ir_labels")
        self.ir_labels = self.hdf5file_ir_labels["ir_labels"]
        self.images = self.hdf5file_train_set["cxr_images"]
        self.cams = self.hdf5file_train_set["cxr_cams"]
        self.base_names = self.hdf5file_train_set["cxr_base_names"]
    
    def __len__(self):
        return self.df.shape[0]
    
    
    def __getitem__(self, index):
        img = torch.FloatTensor(self.images[index])
        base_name = self.base_names[index].decode('utf-8')
        cxr_img = img.permute(1, 2, 0).numpy()
        cxr_img = (cxr_img * 255).round().astype(np.uint8)
        cxr_img, target_scale = imutils.random_scale(cxr_img, scale_range=(0.5, 1.5), order=3)
        cxr_img, box = imutils.random_crop(cxr_img, 320, (0, 255))
        transforms_list = [t.ToTensor(), t.Normalize(mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD)]
        cxr_img = t.Compose([transform for transform in transforms_list])(cxr_img)

        labels = torch.FloatTensor([])
        reduced_labels = torch.FloatTensor([])
        aff_bg_pos_labels = torch.FloatTensor([])
        aff_fg_pos_labels = torch.FloatTensor([])
        aff_neg_labels = torch.FloatTensor([])

        for task_id, task in enumerate(LOCALIZATION_TASKS):
            ir_label = self.ir_labels[index][task_id]
            ir_label, _ = imutils.random_scale(ir_label, scale_range=(0.5, 1.5), order=0, target_scale=target_scale)
            ir_label, _ = imutils.random_crop(ir_label, 320, (0, 255), box=box)
            reduced_label = imutils.pil_rescale(ir_label, 0.25, 0)
            aff_bg_pos_label, aff_fg_pos_label, aff_neg_label = self.extract_aff_lab_func(reduced_label)

            labels = torch.cat((labels, torch.FloatTensor(ir_label).unsqueeze(0)), dim=0)
            aff_bg_pos_labels = torch.cat((aff_bg_pos_labels, torch.FloatTensor(aff_bg_pos_label).unsqueeze(0)), dim=0)
            aff_fg_pos_labels = torch.cat((aff_fg_pos_labels, torch.FloatTensor(aff_fg_pos_label).unsqueeze(0)), dim=0)
            aff_neg_labels = torch.cat((aff_neg_labels, torch.FloatTensor(aff_neg_label).unsqueeze(0)), dim=0)

        out = {
            'img': cxr_img,
            'labels': labels,
            'aff_bg_pos_labels': aff_bg_pos_labels,
            'aff_fg_pos_labels': aff_fg_pos_labels,
            'aff_neg_labels': aff_neg_labels
        }

        return out

# This loader is used to convert the original CheXpert dataset to HDF5 format. Used by the convert_to_hdf5.py script
class CheXpertConvertHDF5DatasetIRNet(Dataset):
    def __init__(self):
        self.csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
        self.img_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
        self.image_level_labels_path = CHEXPERT_UNCERTAIN_DIR / "uncertainty_zeros.csv"
        self.df = pd.read_csv(Path(self.csv_path))
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        base_name = self.df.iloc[index].values[0]
        img_pkl_file = base_name + "_Cardiomegaly_map.pkl"
        cxr_img = pd.read_pickle(self.img_path / img_pkl_file)['cxr_img']

        all_cams = torch.FloatTensor([])
        for task in LOCALIZATION_TASKS:
            cam_pkl_file = base_name + f"_{task}_map.pkl"
            cam = pd.read_pickle(self.img_path / cam_pkl_file)['map']
            all_cams = torch.cat((all_cams, torch.FloatTensor(cam)), dim=0)
        
        return {'img': cxr_img, 
                'base_name': base_name,
                'all_cams': all_cams
                }

class CheXpertHDF5DatasetIRNet(Dataset):
    def __init__(self):
        self.file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_set_ten_percent_combined"
        self.images = self.file["images"]
        self.cams = self.file["cams"]
    
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return {'images': self.images[index], 
                'cams': self.images[index]
                }

# This loader is used to generate pseudo labels from a trained IRNet model and from the CAMs 
class CheXpertMSDatasetIRNet(BaseDataset):
    def __init__(self, csv_name, is_training_set,
                 transform_args, transform, scales=(1.0, 0.75, 0.5, 0.25)):
        super().__init__(csv_name, is_training_set, transform_args)
        self.is_training_set = is_training_set
        if is_training_set:
            self.csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
            self.img_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
            self.hdf5file_train_set = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_set")
            self.gt_seg_labels_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "seg_labels_encoded.json"
            self.image_level_labels_path = CHEXPERT_UNCERTAIN_DIR / "uncertainty_zeros.csv"
            self.image_level_labels_df = pd.read_csv(self.image_level_labels_path)
            self.image_level_labels_df = self.image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})
            with open(self.gt_seg_labels_path) as f:
                self.gt_seg_labels = json.load(f)
            self.df = pd.read_csv(Path(self.csv_path))
        # If test set is being provided, then we need to change some of the paths
        else:
            self.img_path = CHEXPERT_PARENT_TEST_CAMS_DIR / "cams"
            all_pkl_files = list(Path(self.img_path).rglob("*_map.pkl"))
            self.gt_seg_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_encoded.json"
            self.image_level_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_labels.csv"
            self.image_level_labels_df = pd.read_csv(self.image_level_labels_path)
            self.image_level_labels_df = self.image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})
            with open(self.gt_seg_labels_path) as f:
                self.gt_seg_labels = json.load(f)
                img_ids_list = list(self.gt_seg_labels.keys())
                self.df = pd.DataFrame(img_ids_list)

        self.scales = scales
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]

    
    def __getitem__(self, index):
        # Get name of the patient study combination
        base_name = self.df.iloc[index].values[0]
        if self.is_training_set:
            img_level_label_name = "CheXpert-v1.0/train/" + base_name.replace("_", "/", 2) + ".jpg"
        else:
            img_level_label_name = "test/" + base_name.replace("_", "/", 2) + ".jpg"
        img_level_label_row = self.image_level_labels_df.loc[self.image_level_labels_df['Path'] == img_level_label_name]
        img_level_label_row = np.nan_to_num(img_level_label_row[LOCALIZATION_TASKS].values[0])

        # Get the pickle file storing the actual image. 
        # Considering one of the classifications since each of them have the same image
        img_pkl_file = base_name + "_Cardiomegaly_map.pkl"
        cxr_img = pd.read_pickle(self.img_path / img_pkl_file)['cxr_img']
        cxr_img = cxr_img.permute(1, 2, 0).numpy()
        cxr_img = (cxr_img * 255).round().astype(np.uint8)
        transforms_list = [t.ToTensor(), t.Normalize(mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD)]
        cxr_img = t.Compose([transform for transform in transforms_list])(cxr_img)
        size = pd.read_pickle(self.img_path / img_pkl_file)['cxr_dims']
        all_cams = torch.FloatTensor([])
        all_gt_seg_labels = torch.FloatTensor([])
        cxr_img = torch.stack([cxr_img, cxr_img.flip(-1)], dim=0)
  
        for task in LOCALIZATION_TASKS:
            cam_pkl_file = base_name + f"_{task}_map.pkl"
            cam = pd.read_pickle(self.img_path / cam_pkl_file)['map']
            strided_size = imutils.get_strided_size((320, 320), 4)
            cam = F.interpolate(cam, strided_size, mode='bilinear', align_corners=False)[0]
            cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
            all_cams = torch.cat((all_cams, torch.FloatTensor(cam)), dim=0)

            gt_seg_label = mask.decode(self.gt_seg_labels[base_name][task]) * 255
            seg_labels_transforms = t.Compose([t.ToPILImage(), t.Resize((320, 320), interpolation=Image.NEAREST), t.ToTensor()])
            gt_seg_label = seg_labels_transforms(gt_seg_label)
            all_gt_seg_labels = torch.cat((all_gt_seg_labels, torch.FloatTensor(gt_seg_label)), dim=0)

        return {'img': cxr_img, 
                'all_cams': all_cams, 
                'size': (size[0], size[1]), 
                'base_name': base_name, 
                'gt_seg_labels': all_gt_seg_labels,
                'img_level_labels': img_level_label_row}

# This loader is the same as the previous one but it makes use of the hdf5 file formats for loading in
# the train set 
class CheXpertMSDatasetHDF5IRNet(BaseDataset):
    def __init__(self, csv_name, is_training_set,
                 transform_args, transform, scales=(1.0, 0.75, 0.5, 0.25)):
        super().__init__(csv_name, is_training_set, transform_args)
        self.is_training_set = is_training_set
        if is_training_set:
            self.csv_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "train_cams_paths.csv"
            self.img_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "cams"
            self.hdf5file_train_set = h5py.File(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "hdf5_files/train_set")
            self.gt_seg_labels_path = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "seg_labels_encoded.json"
            self.image_level_labels_path = CHEXPERT_UNCERTAIN_DIR / "uncertainty_zeros.csv"
            self.image_level_labels_df = pd.read_csv(self.image_level_labels_path)
            self.image_level_labels_df = self.image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})
            self.images = self.hdf5file_train_set["cxr_images"]
            self.cams = self.hdf5file_train_set["cxr_cams"]
            self.base_names = self.hdf5file_train_set["cxr_base_names"] 
            with open(self.gt_seg_labels_path) as f:
                self.gt_seg_labels = json.load(f)
            self.df = pd.read_csv(Path(self.csv_path))
        else:
            self.img_path = CHEXPERT_PARENT_TEST_CAMS_DIR / "cams"
            all_pkl_files = list(Path(self.img_path).rglob("*_map.pkl"))
            self.gt_seg_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_encoded.json"
            self.image_level_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_labels.csv"
            self.image_level_labels_df = pd.read_csv(self.image_level_labels_path)
            self.image_level_labels_df = self.image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})
            with open(self.gt_seg_labels_path) as f:
                self.gt_seg_labels = json.load(f)
                img_ids_list = list(self.gt_seg_labels.keys())
                self.df = pd.DataFrame(img_ids_list)

        self.scales = scales
        self.transform = transform
    
    def __len__(self):
        return self.df.shape[0]

    
    def __getitem__(self, index):
        # Get name of the patient study combination
        
        base_name = self.base_names[index].decode('utf-8')
        if self.is_training_set:
            img_level_label_name = "CheXpert-v1.0/train/" + base_name.replace("_", "/", 2) + ".jpg"
        else:
            img_level_label_name = "test/" + base_name.replace("_", "/", 2) + ".jpg"
        img_level_label_row = self.image_level_labels_df.loc[self.image_level_labels_df['Path'] == img_level_label_name]
        img_level_label_row = np.nan_to_num(img_level_label_row[LOCALIZATION_TASKS].values[0])

        cxr_img = torch.FloatTensor(self.images[index])
        cxr_img = cxr_img.permute(1, 2, 0).numpy()
        cxr_img = (cxr_img * 255).round().astype(np.uint8)
        transforms_list = [t.ToTensor(), t.Normalize(mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD)]
        cxr_img = t.Compose([transform for transform in transforms_list])(cxr_img)
        all_cams = torch.FloatTensor([])
        all_gt_seg_labels = torch.FloatTensor([])
        cxr_img = torch.stack([cxr_img, cxr_img.flip(-1)], dim=0)

        for task_id, task in enumerate(LOCALIZATION_TASKS):
            cam = torch.FloatTensor(self.cams[index][task_id]).unsqueeze(0)
            strided_size = imutils.get_strided_size((320, 320), 4)
            cam = F.interpolate(cam, strided_size, mode='bilinear', align_corners=False)[0]
            cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
            all_cams = torch.cat((all_cams, torch.FloatTensor(cam)), dim=0)

            gt_seg_label = mask.decode(self.gt_seg_labels[base_name][task]) * 255
            seg_labels_transforms = t.Compose([t.ToPILImage(), t.Resize((320, 320), interpolation=Image.NEAREST), t.ToTensor()])
            gt_seg_label = seg_labels_transforms(gt_seg_label)
            all_gt_seg_labels = torch.cat((all_gt_seg_labels, torch.FloatTensor(gt_seg_label)), dim=0)
        
        return {'img': cxr_img, 
                'all_cams': all_cams, 
                'size': (320, 320), 
                'base_name': base_name, 
                'gt_seg_labels': all_gt_seg_labels,
                'img_level_labels': img_level_label_row}

class CheXpertTestCAMDatasetIRNet(Dataset):
    def __init__(self):
        self.img_path = CHEXPERT_PARENT_TEST_CAMS_DIR / "cams"
        all_pkl_files = list(Path(self.img_path).rglob("*_map.pkl"))
        self.gt_seg_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_encoded.json"
        self.cam_seg_labels_path = CHEXPERT_PARENT_TEST_CAMS_DIR / "seg_labels_encoded.json"
        self.image_level_labels_path = TEST_SEG_LABELS_GT_BASE_PATH / "test_labels.csv"
        self.image_level_labels_df = pd.read_csv(self.image_level_labels_path)
        self.image_level_labels_df = self.image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})
        with open(self.gt_seg_labels_path) as f:
            self.gt_seg_labels = json.load(f)
            img_ids_list = list(self.gt_seg_labels.keys())
            self.df = pd.DataFrame(img_ids_list)
        
        with open(self.cam_seg_labels_path) as f:
            self.cam_seg_labels = json.load(f)

    def __len__(self):
        return self.df.shape[0]

    
    def __getitem__(self, index):
        # Get name of the patient study combination
        base_name = self.df.iloc[index].values[0]
        img_level_label_name = "test/" + base_name.replace("_", "/", 2) + ".jpg"
        img_level_label_row = self.image_level_labels_df.loc[self.image_level_labels_df['Path'] == img_level_label_name]
        img_level_label_row = img_level_label_row[LOCALIZATION_TASKS].values[0]

        all_gt_seg_labels = torch.FloatTensor([])
        all_cams_seg_labels = torch.FloatTensor([])
  
        for task in LOCALIZATION_TASKS:
            gt_seg_label = mask.decode(self.gt_seg_labels[base_name][task]) * 255
            seg_labels_transforms = t.Compose([t.ToPILImage(), t.Resize((320, 320), interpolation=Image.NEAREST), t.ToTensor()])
            gt_seg_label = seg_labels_transforms(gt_seg_label)
            cam_seg_label = mask.decode(self.cam_seg_labels[base_name][task])
            all_gt_seg_labels = torch.cat((all_gt_seg_labels, torch.FloatTensor(gt_seg_label)), dim=0)
            all_cams_seg_labels = torch.cat((all_cams_seg_labels, torch.FloatTensor(cam_seg_label).unsqueeze(0)), dim=0)
        
        
        return {'base_name': base_name, 
                'gt_seg_labels': all_gt_seg_labels,
                'cam_seg_labels': all_cams_seg_labels,
                'img_level_labels': img_level_label_row}




