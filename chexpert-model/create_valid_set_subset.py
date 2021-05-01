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
                        type=int, default=50,
                        required=True,
                        help='Length of subset')                     

    args = parser.parse_args()
    return args

def create_subset(subset_len):
    output_val_set = {}
    seg_labels_file = '/deep/group/aihc-bootcamp-spring2020/localize/annotations/valid_encoded.json'
    curr_pos_images = 0

    with open(seg_labels_file) as f:
        seg_labels = json.load(f)
    
    for key in tqdm(random.sample(seg_labels.keys(), subset_len)):
        output_val_set[key] = seg_labels[key]
        
    with open(f'/deep/group/aihc-bootcamp-spring2020/localize/annotations/valid_encoded_{subset_len}_cxrs.json', "w") as f:
            json.dump(output_val_set, f)

if __name__ == "__main__":
    args = parse_script_args()
    create_subset(args.subset_len)