import random
import json
from constants import *
from pycocotools import mask
from argparse import ArgumentParser
from tqdm import tqdm
import os

def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): Parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--subset_len',
                        type=int, default=100,
                        required=False,
                        help='Length of subset')

    parser.add_argument('--pseudo_labels_type',
                        type=str, default="cams",
                        required=False,
                        help='Type of pseudo label cam/irnet')                        


    args = parser.parse_args()
    return args

def create_subset(subset_len, pseudo_labels_type):
    output_train_set = {}
    irnet_seg_labels_file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / f"pseudo_seg_labels_encoded_{pseudo_labels_type}.json"
    curr_pos_images = 0

    with open(irnet_seg_labels_file) as f:
        seg_labels = json.load(f)
        
    for key in tqdm(random.sample(seg_labels.keys(), subset_len)):
        output_train_set[key] = seg_labels[key]
        
    with open(os.path.join(CHEXPERT_PARENT_TRAIN_CAMS_DIR, f"pseudo_seg_labels_{pseudo_labels_type}_{subset_len}_cxrs.json"), "w") as f:
        json.dump(output_train_set, f)

if __name__ == "__main__":
    args = parse_script_args()
    create_subset(args.subset_len, args.pseudo_labels_type)