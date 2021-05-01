# Helper fuctions to plot segmentations
import glob
from PIL import Image, ImageDraw
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F
from pycocotools import mask
import pandas as pd

# result dir
GROUP_PATH = '/deep/group/aihc-bootcamp-spring2020/localize'

# original x-ray images
CXR_VALID_PATH = '/deep/group/CheXpert/CheXpert-v1.0/valid'
CXR_TEST_PATH = '/deep/group/anujpare/CheXpert_original_test500/test'

# label paths
LABEL_TEST_PATH = '/deep/group/anujpare/CheXpert_original_test500/test.csv'
LABEL_VALID_PATH = '/deep/group/CheXpert/CheXpert-v1.0/valid.csv'


def visualize_segmentations(img_id,task,phase='test',method='vietnam',model='ensemble',w=0.92):
    """
    Visualize the segmentation overlay of stanford annotation + one method of choosing
    
    Args:
        phase(str): choose from 'valid' or 'test'
        method(str): choose from 'gradcam', 'ig' and 'vietnam'
        model(str): choose from 'single' or 'ensemble'. Default for IG is ensemble
        img_id(str): unique image identifier, e.g. patient64741_study1_view1_frontal
        task(str): choose from one of the pathologies
        w(float): opacity of the overlay; 1 for very light segmentation and 0 for very strong
    """
    # load stanford annotation
    stanford_path = f'{GROUP_PATH}/annotations/{phase}_encoded.json'
    gt_segm = load_annotation(stanford_path,img_id,task)
    
    # load segmentation mask from one of the localization methods or human (vietnam)
    if method == 'vietnam':   
        segm_path = f'{GROUP_PATH}/annotations/vietnam_encoded.json'
    elif 'ig' in method or 'nt' in method:
        segm_path = f'{GROUP_PATH}/eval_results/{method}/{phase}_{method}_{model}_encoded_threshold.json'
    else:
        segm_path = f'{GROUP_PATH}/eval_results/{method}/{phase}_{method}_{model}_encoded.json'
    pred_segm = load_annotation(segm_path,img_id,task)
    
    # load original image
    cxr = load_cxr(phase,img_id)

    # create overlay and output the resized image
    overlay = overlay_segmentation(pred_segm,cxr,w)
    overlay = overlay_segmentation(gt_segm,overlay,w,color = [255, 0, 0])
    output = Image.fromarray(overlay).resize((int(cxr.shape[1]/5),int(cxr.shape[0]/5)))
    
    return output


def display_segmentations(img_id,task,phase='test',method='vietnam',model='ensemble',w=0.2, color = [255, 0, 0]):
    """
    Dislay segmentation overlay one at a time
    
    Args:
        phase(str): choose from 'valid' or 'test'
        method(str): choose from 'gradcam', 'ig' and 'vietnam'
        model(str): choose from 'single' or 'ensemble'. Default for IG is ensemble
        img_id(str): unique image identifier, e.g. patient64741_study1_view1_frontal
        task(str): choose from one of the pathologies
        w(float): opacity of the overlay; 1 for very light segmentation and 0 for very strong
    """
    # load original image
    cxr = load_cxr(phase,img_id)
    
    if method == 'stanford':
        # load stanford annotation
        stanford_path = f'{GROUP_PATH}/annotations/{phase}_encoded.json'
        segm = load_annotation(stanford_path,img_id,task)
    
    # load segmentation mask from one of the localization methods or human (vietnam)
    elif method == 'vietnam':   
        segm_path = f'{GROUP_PATH}/annotations/vietnam_encoded.json'
        segm = load_annotation(segm_path,img_id,task)
    else:
        segm_path = f'{GROUP_PATH}/eval_results/{method}/{phase}_{method}_{model}_encoded.json'
        segm = load_annotation(segm_path,img_id,task)

    # create overlay and output the resized image
    overlay = overlay_segmentation(segm,cxr,w,color = color)
    output = Image.fromarray(overlay).resize((int(cxr.shape[1]/5),int(cxr.shape[0]/5)))
    
    output.save(f'localization_eval/fig1/{task}_{method}_{img_id}.png')
    return output


def load_all_ids(phase='test',task=None,pos_only=False):
    """
    Load all image ids. Can choose to only return image ids with positive labels of a given task
    
    Args:
        phase(str): choose 'valid' or 'test'
        task(str): one of the localization task
        pos_only(bool): True if only return positive instances
    
    Returns:
        all_ids(list): a list of all image ids to be returned
    """
    path = LABEL_VALID_PATH if phase == 'valid' else LABEL_TEST_PATH
    labels = pd.read_csv(path)
    
    # resolve naming inconsistency
    if task is not None:
        task = task.replace('Airspace Opacity','Lung Opacity')

    if pos_only: 
        labels = labels[labels[task] == 1]
    
    all_ids = labels.Path.map(lambda x: '_'.join(x.split('/')[2:]).replace('.jpg','')).tolist()
    
    return all_ids

def load_annotation(path,patientid,task):
    """
    Load annotations from json path. 
    Args:
        path(str): json path
        patientid(str): patient id
        task(str): one of the localization pathology
    
    Returns:
        segm_map(numpy.ndarray,np.uint8): np array of the segmentation mask
    """
    with open(path) as f:
        ann = json.load(f)
    
    if patientid in ann:
        ann_item = ann[patientid][task]
        segm_map = mask.decode(ann_item)
    else:
        print('no pathologies labeled in the case')
        return None
    return segm_map


def load_cxr(phase,img_id):
    """
    Load original chest x-ray images given unique image identifier (used in evaluation)
    
    Args:
        phase(str): 'valid' or 'test'
        image_id(str): unique image identifier
    
    Returns:
        cxr(numpy.ndarray): original chest x-ray image
    """
    img_dir = CXR_TEST_PATH if phase == 'test' else CXR_VALID_PATH
    names = img_id.split('_') 
    assert len(names) == 4
    patientid = names[0]
    study = names[1]
    img_name = '_'.join(names[2:]) 
    cxr = plt.imread(f'{img_dir}/{patientid}/{study}/{img_name}.jpg')
    
    return cxr
    
    
def overlay_segmentation(mask,cxr,w,color=[255, 255, 0]):
    """
    Overlay sementation on original chest x-ray image
    
    Args:
        cxr (numpy.ndarray,np.uint8): grayscale segmentation image of original x-ray size
        mask (numpy.ndarray,np.uint8): grayscale image of original x-ray 
    Returns:
        overlay(numpy.ndarry): overlay of the original image size
    """
    if len(cxr.shape) ==2:
        rgb_cxr = cv2.cvtColor(cxr, cv2.COLOR_GRAY2RGB)
    
    else:
        rgb_cxr = cxr
    
    if mask is not None:
        im_colored = mask.astype(np.uint8)*255
        im_rgb = cv2.cvtColor(im_colored, cv2.COLOR_GRAY2RGB)
        colored = np.where(im_rgb == [0,0,0], im_rgb, color)
        overlay = cv2.addWeighted(rgb_cxr,0.92,colored.astype(np.uint8),1-w,0)
    else:
        overlay = rgb_cxr
    
    return overlay


# helper function that merges test_image_paths.csv and test_groundtruth.csv
def create_test_label():
    """
    Note:
        valid.csv has both unique image identifier and ground truth labels whereas test_groundtruth.csv 
        has only study name (not image unique). Therefore we did some extra processing.
    """
    label_path = '/deep/group/anujpare/CheXpert_original_test500/test_groundtruth.csv'
    img_path = '/deep/group/anujpare/CheXpert_original_test500/test_image_paths.csv'
    
    studies = pd.read_csv(label_path)
    imgs = pd.read_csv(img_path)
    
    # extract identidier that matches test_groundtruth's 'Study' variable
    imgs['Study'] = imgs['Path'].map(lambda x: '/'.join(x.split('/')[:4]))
    
    # join two datasets
    test_labels = pd.merge(imgs,studies,on='Study',how='left')
    
    # write to csv
    test_labels.to_csv('/deep/group/anujpare/CheXpert_original_test500/test.csv', index = False)
    