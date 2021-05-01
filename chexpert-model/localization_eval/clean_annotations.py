import json
import pandas as pd
import argparse
from pathlib import Path

path_group = '/deep/group/aihc-bootcamp-spring2020/localize'

def clean_label(dataset = 'valid'):
    """
    Rewrite md.ai labels to an organized format
    """    
    label_file = f'{path_group}/annotations/val_test_mdai_stanford_project_7.json'
    
    name_mapping_file = f'{path_group}/annotations/mdai_stanford_project_7_mapping_{dataset}.csv'
    write_file = f'{path_group}/annotations/{dataset}_annotations.json'
    
    if not Path(label_file).exists():
        print("Annotations file not found")
        return
    
    if not Path(name_mapping_file).exists():
        print("Annotations file not found")
        return
    
    print(f'Reading annotations from {label_file}')
    with open(label_file) as f:
        chexpert_valid = json.load(f)
    
    # dictionary for pathology labels
    label_dict = {}
    for label in chexpert_valid['labelGroups'][0]['labels']:
        label_id = label['id']
        name = label['name']
        name = 'Support Devices' if 'Support Device' in name else name
        name = 'Airspace Opacity' if 'Lung Opacity' in name else name
        label_dict[label_id] = name
        
    # all annotations
    if dataset == 'valid':
        annotations = chexpert_valid['datasets'][0]['annotations']
    elif dataset == 'test':
        annotations = chexpert_valid['datasets'][1]['annotations']
    
    print('Transform names')
    # map from md ai id to image names 
    name_mapping = pd.read_csv(name_mapping_file, sep=",")
    if dataset == 'valid':
        name_mapping['name'] = name_mapping['original_filename'].apply(lambda x: x.split('/')[1].replace(".jpg",'').replace('localize_','').replace('view','_view'))
    else:
        name_mapping['name'] = name_mapping['original_filename'].apply(lambda x: x.split('/')[1].replace(".jpg",'').replace('tlocalize_','').replace('view','_view'))
    name_dict = dict([(Id,name) for name,Id in zip(name_mapping.name, name_mapping.StudyInstanceUID)])

    ground_truth = {}
    
    print(f'Write to cleaned format at {write_file}')
    # write to new format
    not_found = 0
    for item in annotations:
        
        if item['StudyInstanceUID'] not in name_dict:
            not_found += 1
            continue
        
        name = name_dict[item['StudyInstanceUID']]

        label = label_dict[item['labelId']]
        polygon = item['data']['vertices']

        if name not in ground_truth:
            ground_truth[name] = {}
        
        # add image size
        ground_truth[name]['img_size'] = (item['height'],item['width'])
        
        # add pathology contour coordinates
        if label in ground_truth[name]:
            ground_truth[name][label].append(polygon)
        else:
            ground_truth[name][label]= [polygon]
    
    
#     for study_id in name_dict:
#         name = name_dict[study_id]
#         if name not in ground_truth:
#             ground_truth[name] = {}
    print(f"Writing cleaned annotations to {write_file}")
    with open(write_file, "w") as outfile:  
        json.dump(ground_truth, outfile) 
    
    return write_file


def clean_vietnam():
    """
    Clean vietnam annotations and export to json
    
    The output json is organized such that:
    
    patientid:
        tasks1:
            polygons coordinates 
        task2:
            ...
    
    Only the positive labels appear in the annotation data
    """
    vietnam_file = f'{path_group}/annotations/test_vietnam.json'
    output_path = f'{path_group}/annotations/vietnam_annotations.json'
    
    print(f"Loading annotation file from {vietnam_file}")
    with open(vietnam_file) as f:
        chexpert_vietnam = json.load(f)
    
    print("Reading labels")
    label_dict = {}
    for idx,label in enumerate(chexpert_vietnam['labelGroups']['0']['labels']):
        label_id = idx
        name = label['name']
        name = 'Support Devices' if 'Support Device' in name else name
        name = 'Airspace Opacity' if 'Lung Opacity' in name else name
        label_dict[label_id] = name   
    
    annotations = chexpert_vietnam['datasets']['0']['annotations']
    
    print("Create clean annotation file")
    vietnam_ann = {}
    
    for item in annotations:
        
        name = '_'.join(item['fileId'].replace('.jpg','').split('/')[1:])   
        label = label_dict[item['labelId']]
        img_size = [item['height'],item['width']]

        if name not in vietnam_ann:
            vietnam_ann[name] = {}

        if 'img_size' not in vietnam_ann[name]:
            vietnam_ann[name]['img_size'] = img_size

        for instance in item['data']:

            polygon = instance['vertices']

            if label in vietnam_ann[name]:
                vietnam_ann[name][label].append(polygon)
            else:
                vietnam_ann[name][label]= [polygon]
    
    print(f'Writing to new json at {output_path}')
    with open(output_path, "w") as outfile:  
        json.dump(vietnam_ann, outfile)
        
    return output_path


def merge_ann(ann_file):
    """
    Merge Cardiomegaly and Enlarged Cardiomediastinum annotations for the label Cardiomediastinum
    
    Args:
        ann_file(str): input annotation json file
    Returns:
        None
        (Save merged file to original location)
    """
    print(f'Read original annotations from {ann_file}')
    with open(ann_file) as f:
        ann = json.load(f)
        
    print('Merge labels')
    for img_id in ann.keys():
        if 'Cardiomegaly' in ann[img_id]:
        
            if 'Enlarged Cardiomediastinum' in ann[img_id]:
                ann[img_id]['Enlarged Cardiomediastinum'] += ann[img_id]['Cardiomegaly']
            else:
                ann[img_id]['Enlarged Cardiomediastinum'] = ann[img_id]['Cardiomegaly']
    
    write_file = ann_file.replace('.json','_merged.json')
    print(f'Write to new file at {write_file}')
    with open(write_file, "w") as outfile:  
        json.dump(ann, outfile) 
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'valid' , help = "valid, test or vietnam")
    args = parser.parse_args()
    
    if args.dataset == "vietnam":
        output_path = clean_vietnam()
    else:
        output_path = clean_label(args.dataset)
    
    merge_ann(output_path)
    
    