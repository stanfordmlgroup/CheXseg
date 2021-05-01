import torch.utils.data as data
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as albu

from .distillation_dataset import DistillationDataset
from constants import *

import numpy as np

def get_dist_loader(model, masks_path, encoder, encoder_weights, data_args, transform_args, model_args, is_training, images_path=None, logger=None):
    """Get PyTorch data loader.

    Args:
        model: teacher model to find predicted outputs
        masks_path: path of saved teacher predictions (masks), or path to save predictions
        encoder: teacher model encoder
        encoder_weights: teacher model encoder weights
        data_args: Namespace of data arguments.
        transform_args: Namespace of transform arguments.
        model_args: Namespace of model arguments
        is_training: Bool indicating whether in training mode.
        logger: Optional Logger object for printing data to stdout and file.

    Return:
        loader: PyTorch DataLoader object
    """

    shuffle = is_training
    num_workers = data_args.num_workers
    batch_size = data_args.batch_size

    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_args.encoder, model_args.encoder_weights)
    dataset = DistillationDataset(model, masks_path, encoder, encoder_weights, transform_args.scale, data_args.classes, images_path=images_path, preprocessing=get_preprocessing(preprocessing_fn))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)