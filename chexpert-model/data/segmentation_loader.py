import torch.utils.data as data
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as albu

from .segmentation_dataset import SegmentationDataset, SemiSupervisedWeightedDataset
from constants import *

def get_seg_loader(phase, data_args, transform_args, model_args,
                   is_training, logger=None, semi_supervised=False, include_mask=True):
    """Get PyTorch data loader.

    Args:
        phase: string name of training phase {train, valid, test}.
        data_args: Namespace of data arguments.
        transform_args: Namespace of transform arguments.
        model_args: Namespace of model arguments
        is_training: Bool indicating whether in training mode.
        logger: Optional Logger object for printing data to stdout and file.
        semi_supervised: Indicate whether to use both pseudo-labels and expert annotations

    Return:
        loader: PyTorch DataLoader object
    """
    shuffle = is_training
    num_workers = data_args.num_workers
    batch_size = data_args.batch_size
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_args.encoder, model_args.encoder_weights)
    # print("Semi-supervised=", semi_supervised)
    if semi_supervised and is_training:
        assert data_args.ss_expert_annotations_masks_path and data_args.ss_dnn_generated_masks_path
        fs_dataset = SegmentationDataset('valid', data_args.ss_expert_annotations_masks_path, transform_args.scale, data_args.classes, preprocessing=get_preprocessing(preprocessing_fn))
        ws_dataset = SegmentationDataset(phase, data_args.ss_dnn_generated_masks_path, transform_args.scale, data_args.classes, preprocessing=get_preprocessing(preprocessing_fn))
        print(f"Semi-supervised: using {len(fs_dataset)} strongly labeled images and {len(ws_dataset)} weakly labeled images.")
        if data_args.weighted:
            dataset = SemiSupervisedWeightedDataset(fs_dataset, ws_dataset, data_args)
        else:
            datasets = [fs_dataset, ws_dataset]
            dataset = data.ConcatDataset(datasets)
    else:
        if is_training:
            masks_path = data_args.train_masks_path 
        else:
            if semi_supervised:
                masks_path = data_args.eval_masks_path
            else:
                masks_path = data_args.eval_masks_path

        assert masks_path
        dataset = SegmentationDataset(phase, masks_path, transform_args.scale, data_args.classes, preprocessing=get_preprocessing(preprocessing_fn), include_mask=include_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

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
