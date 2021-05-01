import numpy as np
from pycocotools import mask
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
import json
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from eval_constants import *

def create_mask(polygons,img_dims):
    """
    Creates a binary mask (of the original matrix   size) given a list of polygons
    
    Args:
        poly_coords (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]
    
    Returns:
        mask (np.array): 1 where the pixel is predicted to be the pathology 0 otherwise
    """
    poly = Image.new('1',(img_dims[1],img_dims[0]))
    for polygon in polygons:
        coords = [(point[0],point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords,  outline = 1, fill = 1)
    
    mask = np.array(poly,dtype = "int")
    return mask

def cam_to_segmentation(cam_mask, smoothing = False, k = 5, override_negative = False):
    """
    Convert CAM heatmap to binary segmentation mask
    
    Args:
        cam_mask: heatmap of the original image size. dim: H x W. Will squeeze the tensor if there are more than two dimensions
    
    Returns:
        segmentation_output (PIL Image): binary segmentation output
    """
    if (len(cam_mask.size()) > 2):
        cam_mask = cam_mask.squeeze()
    
    assert len(cam_mask.size()) == 2
    
    if override_negative:
        img_dims = (cam_mask.size()[1],cam_mask.size()[0])
        return Image.new('1',img_dims)
    
    if smoothing:
        mask = cam_mask - cam_mask.min()
        mask = 255 * mask.div(mask.max()).data
        mask = np.uint8(mask)
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        ksize = int(mask.shape[0]/k)
        if ksize % 2 ==0:
            ksize += 1
        gray_img = cv2.boxFilter(cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY),-1,(ksize,ksize))
        gray_img = 255-gray_img
    else:
#         normalize the CAM before segmentation
#         mask = cam_mask - cam_mask.min()
#         mask = 255 * mask.div(mask.max()).data
#         mask = np.uint8(mask)
        mask = cam_mask.cpu().detach().numpy()
        gray_img = (mask*255).astype(np.uint8)
    
    # find contours to generate segmentation
    maxval = np.max(gray_img)
    thresh = cv2.threshold(gray_img, 0, maxval, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    polygons = []
    for cnt in cnts:
        if len(cnt) >1:
            polygons.append([list(pt[0]) for pt in cnt])            
    
    img_dims = (mask.shape[1],mask.shape[0])
    segmentation_output = Image.new('1',img_dims)
    
    for polygon in polygons:
        coords = [(point[0],point[1]) for point in polygon]
        ImageDraw.Draw(segmentation_output).polygon(coords,  outline = 1, fill = 1)
       
    return segmentation_output

def segmentation_to_png(save_dir,segmentation_mask, task,img_name):
    """
    Save segmentation map to pngs to a structured folder
    
    Args:
        save_dir: directory to save png results to 
        segmentation_mask (PIL Image): binary segmentation mask
        task (str): name of the pathology
        img_name (str): patient name
    """
    save_path = '/'.join([save_dir] + img_name.split('_')[:-2])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    result_name = '_'.join([task] + img_name.split('_')[-2:]) + '.png'
    segmentation_mask.save(save_path + '/' + result_name)

def segmentation_to_mask(label_mask):
    """
    Encodes a segmentation mask (of a given pathology) using the Mask API.
    Args:
        label_mask: [h x w] binary segmentation mask that indicates the label of each pixel 
    
    Returns:
        Rs - the encoded label mask for label 'label_id'
    """

    label_mask = np.asfortranarray(label_mask.astype('uint8'))
    Rs = mask.encode(label_mask)
    Rs['counts'] = Rs['counts'].decode()
    return Rs
            
        
def gt_to_png(gt_file,output_dir):
    """
    Take the ground truth contour data, convert them to segmentation pngs and save to a given directory
    
    Args:
        gt_file (json): processed ground truth file (json)
        output_dir (str): directory to save the pngs 
    """
    
    with open(gt_file) as f:
        gt = json.load(f)

    
    for patient in gt.keys():
        for task in ALL_TASKS:
            
            print("Starting " + task)
            # create a blank image
            img_dims = gt[patient]['img_size']
            truth = Image.new('1',img_dims)
            
            # change naming convention
            gt_task = task.replace('Airspace', 'Lung').replace('Devices','Device')
            
            if gt_task in gt[patient].keys():
                polygons = gt[patient][gt_task]
                for polygon in polygons:
                    coords = [(point[0],point[1]) for point in polygon]
                    ImageDraw.Draw(truth).polygon(coords,  outline = 1, fill = 1)
            
            # save to directory
            segmentation_to_png(output_dir,truth, task,patient)
            
            