import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as t
import wandb
from .model_util import get_probs, uncertain_logits_to_probs
from cams.gradcam_nt import gradcam_nt
from cams.integrated_gradients import integrated_gradient
from constants import *
from PIL import Image
from PIL.PngImagePlugin import Image, PngInfo
from torchvision.utils import save_image
from util.model_util import get_model_folder


def get_heatmap(mask):
    """Make heatmap from mask.
    
    Args:
        mask (torch.tensor): mask with shape (1, 1, H, W)
        
    Return:
        heatmap (torch.tensor): heatmap image with shape (3, H, W)
    """
    # normalize mask
    mask = mask - mask.min()
    mask = 255 * mask.div(mask.max()).data
    
    heatmap = cv2.applyColorMap(np.uint8(mask.detach().squeeze().cpu()), cv2.COLORMAP_JET)
    
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    return heatmap


def get_dimming_factor(prob):
    """Calculate factor by which to multiply heatmap before it's overlaid on CXR.
    
    A higher predicted probability will lead to a brighter heatmap and a lower
    predicted probability will lead to a dimmer heatmap (e.g. a predicted
    probability of 1.0 will return discrete_val=1.0 so that the heatmap will
    not be dimmed at all; a predicted probability of 0.0 will return
    discrete_val=0.0 so that the heatmap will not be at all visible on the
    overlay; a predicted probability of 0.4 will return discrete_val=0.5
    so that the heatmap is somewhat dimmed).
    
    Args:
        prob (float): predicted probability
        
    Return:
        discrete_val (numpy.float64): float btwn 0 and 1 by which to multiply heatmap
    """
    intervals = np.array([0, 0.2, 0.4, 0.8, 1])
    
    # find index of smallest interval >= prob
    idx_interval = np.where(intervals >= prob)[0][0]
    
    # restrict idx_interval so that it's btwn 0 and 1
    discrete_val = .25 * idx_interval
    
    return discrete_val


def get_labels():
    return [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Lesion",
        "Airspace Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices"
    ]


def get_overlay(heatmap, image, class_idx, prob):
    """Synthesize saliency map overlay using normalized heatmap and unnormalized image.
    
    Args:
        heatmap (torch.tensor): heatmap image with shape (3, H, W)
        image (torch.tensor): image shape of (3, H, W)
        class_idx (int): class index
        prob (float): predicted probability for class_idx
        
    Return:
        overlay (torch.tensor): input image with heatmap overlay with shape (3, H, W)
    """
    overlay = image + heatmap * get_dimming_factor(prob)
    overlay = overlay * 255
    overlay = overlay.div(overlay.max()).squeeze()
    return overlay


def localize(model, model_args, logger_args, transform_args,
             image, img_name, cxr_dims, y, save_cams_tasks):
    model_folder = get_model_folder(model_args.ckpt_path)
    save_path = pathlib.Path(logger_args.save_dir_predictions)/model_folder/'cams'
    os.makedirs(save_path, exist_ok=True)
    
    mean, std = get_normalization_factors(transform_args.normalization)
    cxr_unnorm = unnormalize(image, mean, std).squeeze().detach().cpu()
    
    feature_maps = [] # hold tuples (feature_map, prob, class_idx)
    if logger_args.cam_method == "wildcat":
        x = model.model.features(image)
        sub_map = model.model.classifier(x)
        feature_map = model.spatial_pooling.class_wise(sub_map)
        logits = model.spatial_pooling.spatial(feature_map)
        prob = get_probs(logits, model_args.model_uncertainty)
        # FIXME: Make sure this works for Wildcat when specifying tasks to save
        for i in range(len(save_cams_tasks)):
            feature_maps.append((feature_map[:, i, :, :].unsqueeze(0), float(prob[0, i]), i))
    elif logger_args.cam_method == "ig":
        for task in save_cams_tasks:
            class_idx = CHEXPERT_TASKS.index(task)
            feature_map, logits, prob = integrated_gradient(image, model, model_args, task, n_steps= N_STEPS_IG, 
                                          internal_batch_size = INTERNAL_BATCH_SIZE_IG, 
                                          noise_tunnel_flag = logger_args.noise_tunnel, 
                                          stdevs = logger_args.add_noise, n_samples = N_SAMPLES_NT)
        
            feature_maps.append((feature_map, prob, class_idx))
    else: # defaults to Grad-CAM
        for task in save_cams_tasks:
            class_idx = CHEXPERT_TASKS.index(task)
            feature_map, logits, prob = gradcam_nt(image, model, model_args, task,
                                                   n_samples=N_SAMPLES_NT,
                                                   stdevs=DEFAULT_STDEV_NT,
                                                   noise_tunnel_flag=logger_args.noise_tunnel)
            feature_maps.append((feature_map, prob, class_idx))

    for f, p, c in feature_maps:
        f = F.interpolate(f, size=transform_args.scale, mode='bilinear', align_corners=False)
        task = CHEXPERT_TASKS[c]
        task_y = int(y[c])
        img = {'map': f,
               'prob': p,
               'task': task,
               'gt': task_y,
               'cxr_img': cxr_unnorm,
               'cxr_dims': cxr_dims}
        saliency_map_path = save_path/f"{img_name}_{task}_map.pkl"
        file = open(saliency_map_path, 'wb')
        pickle.dump(img, file)

def post_process_cams(args, save_dir_predictions):
    print("Creating and saving cam heatmaps and overlays...")
    cxrs_path = save_dir_predictions/'cxrs'
    os.makedirs(cxrs_path, exist_ok=True)
    
    cam_paths = [x for x in glob.glob(str(save_dir_predictions/'cams/*.pkl'))]
    cxrs_saved = set()
    for cam_path in cam_paths:
        cam_name = os.path.basename(cam_path)
        cam_name = '_'.join(cam_name.split('_')[:4])
        cam_info = pickle.load(open(cam_path,'rb'))
        p = cam_info['prob']
        task = cam_info['task']
        gt_y = cam_info['gt']
        cxr_unnorm = cam_info['cxr_img']
        cxr_dims = cam_info['cxr_dims']
        c = CHEXPERT_TASKS.index(cam_info['task'])
        
        # save original cxr if it hasn't already been saved for this patient
        if cam_name not in cxrs_saved:
            cxr = F.interpolate(cxr_unnorm.unsqueeze(0), size=cxr_dims,
                                mode='bilinear', align_corners=False)
            cxr = cxr.squeeze(0).detach().cpu()
            if args.logger_args.save_cams_wandb:
                wandb.log({f"{cam_name}": [wandb.Image(t.ToPILImage()(cxr), caption=f"{cam_name}_cxr")]})
            save_path = str(cxrs_path/f"{cam_name}_cxr.png")
            save_image(cxr, save_path)
            cxrs_saved.add(cam_name)
        
        # save heatmap and overlay
        heatmap = get_heatmap(cam_info['map'])
        overlay = get_overlay(heatmap, cxr_unnorm, c, p)
        for img_name, img in {'heatmap': heatmap, 'overlay': overlay}.items():
            img = F.interpolate(img.unsqueeze(0), size=cxr_dims, mode='bilinear', align_corners=False)
            img = img.squeeze(0)
            if args.logger_args.save_cams_wandb:
                wandb.log({f"{cam_name}": [wandb.Image(t.ToPILImage()(img.cpu()),
                                                       caption=f"{task}_prob: {round(p, 3)}_gt: {gt_y}_{img_name}")]})
            img_path = str(cxrs_path/f"{cam_name}_{task}_{img_name}.png")
            save_image(img, img_path)


def get_normalization_factors(normalization):
    if normalization == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif normalization == 'chexpert_norm':
        mean = CHEXPERT_MEAN
        std = CHEXPERT_STD
    else:
        raise ValueError(f"Normalization {normalization} not supported.")

    return mean, std
    

def unnormalize(normalized_imgs, mean, std):
    """Unnormalize image for visualization.
    
    Args:
        normalized_imgs (torch.tensor): image with shape (b, 3, h, w)
        mean (list of floats): list of ImageNet means for 3-channel RGB images
        std (list of floats): list of ImageNet standard deviations for 3-channel RGB images
    
    Returns:
        unnormalized_imgs (torch.tensor): unnormalized image with shape (b, 3, h, w)
    """
    assert(len(normalized_imgs.shape) == 4), f"Image tensor should have 4 dimensions, \
        but got tensor with {len(normalized_imgs.shape)} dimensions."
        
    # make a copy to avoid unnormalizing in place
    normalized_imgs_copy = normalized_imgs.clone()
    r, g, b = normalized_imgs_copy.split(1, dim=1)
    r.mul_(std[0]), g.mul_(std[1]), b.mul_(std[2])
    r.add_(mean[0]), g.add_(mean[1]), b.add_(mean[2])
    unnormalized_imgs = torch.cat([r, g, b], dim=1)

    return unnormalized_imgs
