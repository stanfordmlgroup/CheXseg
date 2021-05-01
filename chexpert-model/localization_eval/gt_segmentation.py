# Create segmentation for ground truth annotations 
# Input: {valid/test/vietnam}_annotations_merged.json
# Output: 1. pngs of the segmentation 2. encoded mask in json format
import json
from eval_helper import create_mask, segmentation_to_mask
import argparse

LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Support Devices"
                  ]


def gt_to_mask(gt_file,output_file):
    """
    Create segmentation based on radiologist-labeled contour polygons 
    """
    print(f"Read ground truth labels from {gt_file}")
    with open(gt_file) as f:
        gt = json.load(f)
    
    print(f"Create segmentation and encode")
    results = {}
    for img_id in gt.keys():
        
        if img_id not in results:
            results[img_id] = {}
        for task in LOCALIZATION_TASKS:
            # create segmentation
            polygons = gt[img_id][task] if task in gt[img_id] else []
            img_dims = gt[img_id]['img_size']
            segm_map = create_mask(polygons,img_dims)  #np array
            
            # encode to coco mask
            encoded_map = segmentation_to_mask(segm_map)
            results[img_id][task] = encoded_map
    
    assert len(results.keys()) == len(gt.keys())
    
    print(f"Write results to json at {output_file}")
    # write results to json file
    with open(output_file, "w") as outfile:  
        json.dump(results, outfile)
    
    
def gt_to_png():
    """
    Create segmentation based on radiologist ground truth and save them to pngs
    """


        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'valid' , help = "valid, test or vietnam")
    args = parser.parse_args()
    
    dataset = args.dataset
    path_group = '/deep/group/aihc-bootcamp-spring2020/localize'
    gt_file = f'{path_group}/annotations/{dataset}_annotations_merged.json'
    gt_encoded_file = f'{path_group}/annotations/{dataset}_encoded.json'
    
    gt_to_mask(gt_file,gt_encoded_file)