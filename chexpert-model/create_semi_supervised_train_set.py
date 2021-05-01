from constants import *
from pycocotools import mask
import json
import pandas as pd
import os
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): Parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--task_to_filter',
                        type=str, default="Enlarged Cardiomediastinum",
                        required=True,
                        help='Localization task to be filtered')

    parser.add_argument('--max_pos_images',
                        type=int, default=10,
                        required=True,
                        help='Maximum number of positively labeled images to be included in the train set')


    args = parser.parse_args()
    return args

def filter_train_set(task_to_filter, max_pos_images):
    image_level_labels_path = CHEXPERT_UNCERTAIN_DIR / "uncertainty_zeros.csv"
    image_level_labels_df = pd.read_csv(image_level_labels_path)
    image_level_labels_df = image_level_labels_df.rename(columns={"Lung Opacity": "Airspace Opacity"})

    if not os.path.exists(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "semi_supervised_train_sets"):
        os.mkdir(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "semi_supervised_train_sets")

    output_train_set_filtered = {}
    irnet_seg_labels_file = CHEXPERT_PARENT_TRAIN_CAMS_DIR / "pseudo_seg_labels_encoded_cams.json"
    curr_pos_images = 0

    with open(irnet_seg_labels_file) as f:
        irnet_seg_labels = json.load(f)
    
    for base_name in tqdm(irnet_seg_labels.keys()):
        img_level_label_name = "CheXpert-v1.0/train/" + base_name.replace("_", "/", 2) + ".jpg"
        img_level_label_row = image_level_labels_df.loc[image_level_labels_df['Path'] == img_level_label_name]
        img_level_label = np.nan_to_num(img_level_label_row[task_to_filter].values[0])

        if img_level_label == 1 and curr_pos_images < max_pos_images:
            output_train_set_filtered[base_name] = irnet_seg_labels[base_name]
            curr_pos_images += 1

        if img_level_label == 0:
            output_train_set_filtered[base_name] = irnet_seg_labels[base_name]
        
    with open(os.path.join(CHEXPERT_PARENT_TRAIN_CAMS_DIR / "semi_supervised_train_sets", f"pseudo_seg_labels_cams_{task_to_filter}_pos_images_{max_pos_images}.json"), "w") as f:
            json.dump(output_train_set_filtered, f)


if __name__ == "__main__":
    args = parse_script_args()
    filter_train_set(args.task_to_filter, args.max_pos_images)
