from args import SegTestArgParser
import segmentation_models_pytorch as smp
from data import get_seg_loader
import torch
import pandas as pd
import util
import json
from argparse import Namespace
import albumentations as albu
import numpy as np
from data import SegmentationDataset

def test(args):
    train_args = args.model_args.config_path + '/args.json'
    ckpt_path = args.model_args.config_path + '/best_model.pth'
    with open(train_args) as f:
        train_args = json.load(f, object_hook=dict_to_namespace)
    model_fn = util.get_seg_model_fn(train_args.model_args.architecture)
    model = model_fn(encoder_name=train_args.model_args.encoder, 
                     encoder_weights=train_args.model_args.encoder_weights, 
                     classes=len(train_args.data_args.classes), 
                     activation=None)
    model.load_state_dict(torch.load(ckpt_path))
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(train_args.model_args.encoder, train_args.model_args.encoder_weights)
    test_dataset = SegmentationDataset('test', None, train_args.transform_args.scale, train_args.data_args.classes, preprocessing=get_preprocessing(preprocessing_fn))

    img_id = "patient65106_study1_view2_lateral"
    task = "Atelectasis" 
    out_path = "/deep/group/aihc-bootcamp-fall2020/objectefficiency/semi_supervised/distillation_teacher_pred"
    img_index = test_dataset.img_ids.index(img_id)
    image, gt_mask = test_dataset[img_index]

    x_tensor = torch.from_numpy(image).unsqueeze(0)

    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().numpy())

    pr_mask = pr_mask[train_args.data_args.classes.index(task)]
    np.save(out_path, pr_mask)
    print('Image saved at', out_path)

def dict_to_namespace(d):
    return Namespace(**d)

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

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

if __name__ == "__main__":
    parser = SegTestArgParser()
    test(parser.parse_args())