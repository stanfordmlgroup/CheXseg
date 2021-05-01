"""Entry-point script to test models."""
import glob
import json
import os
import pandas as pd
import pickle
import shutil
import torch
import wandb
from args import TestArgParser
from constants import CHEXPERT_TASKS
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from train import Model
from util.localization_util import post_process_cams
from util.model_util import get_model_folder


def test_single_model(args, logger, tasks):
    # train_args = torch.load(args.model_args.ckpt_path)['hyper_parameters']
    # train_args_ns = Namespace(**train_args)
    # print(train_args_ns)
    model = Model.load_from_checkpoint(args.model_args.ckpt_path)
    print(model)
    # model = Model(args)
    model.save_cams_tasks = tasks
    logger.watch(model, log='all')

    # TODO: Find another way to pass in test args.
    logger_args = vars(model.logger_args)
    logger_args['save_cams'] = args.logger_args.save_cams
    logger_args['save_train_cams'] = args.logger_args.save_train_cams
    logger_args['save_dir_predictions'] = args.logger_args.save_dir_predictions
    logger_args['cam_method'] = args.logger_args.cam_method
    logger_args['noise_tunnel'] = args.logger_args.noise_tunnel
    logger_args['add_noise'] = args.logger_args.add_noise
    
    data_args = vars(model.data_args)
    data_args['batch_size'] = args.data_args.batch_size
    data_args['phase'] = args.data_args.phase
    
    model_args = vars(model.model_args)
    model_args['ckpt_path'] = args.model_args.ckpt_path
    model_args['config_path'] = args.model_args.config_path

    trainer = Trainer(gpus=[0],
                      logger=logger)
    wandb.config.update({"phase": args.data_args.phase})
    trainer.test(model)
    
    model_folder = get_model_folder(model_args['ckpt_path'])
    save_dir_predictions = Path(logger_args['save_dir_predictions'])/model_folder
    return save_dir_predictions


def aggregate_ckpts(args, save_dir_all_ckpts, config):
    """Aggregates predictions across an ensemble of checkpoints.
    If cams were saved, also aggregate the cams across an ensemble of checkpoints.
    Aggregated predictions (and cams, if applicable) are saved to the path
    indicated in the command line arguments.
    
    Args:
        save_dir_all_ckpts (list): list of directories, one for each checkpoint.
                                   Each directory contains a predictions.csv file,
                                   and if cams were saved, then it also contains
                                   a folder `cams` that holds all cams.
        config (dict): config file with ensemble of ckpts per task
    """
    save_dir_predictions = Path(args.logger_args.save_dir_predictions)/'ensemble_results'
    os.makedirs(save_dir_predictions, exist_ok=True)
    
    # aggregate predictions
    print(f"Aggregating checkpoints...")
    df_list = []
    for path in save_dir_all_ckpts:
        file_path = path/'predictions.csv'
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    if args.logger_args.aggregation_method == 'mean':
        agg_df = pd.concat(df_list).groupby(level=0).mean()
        study = df_list[0]['Study']
        agg_df.insert(0, 'Study', study)

    file_path = save_dir_predictions/'predictions.csv'
    agg_df.to_csv(file_path, index=False)
    
    # if cams were saved, aggregate cams
    if args.logger_args.save_cams:
        print(f"Aggregating cams...")
        file_names = list(set([os.path.basename(x) for x in glob.glob(str(save_dir_all_ckpts[0].parent/"*/cams/*.pkl"))]))
        for file in file_names: # for each patient_view/task pair
            patient_view = '_'.join(file.split('_')[:4])
            task = file.split('_')[-2]
            
            # get all 10 cams created for patient_view/task pair
            map_list = []
            for ckpt in config["task2models"][task]: # for each ckpt mapped to task
                model_folder = get_model_folder(ckpt['ckpt_path'])
                file_path = Path(args.logger_args.save_dir_predictions)/model_folder/'cams'/file
                saliency_map = pickle.load(open(file_path,'rb'))
                map_list.append(saliency_map)
            
            # take mean of 10 cams
            agg_dict = {}
            if args.logger_args.aggregation_method == 'mean':
                agg_dict['map'] = sum(item['map'] for item in map_list) / len(map_list)
                agg_dict['prob'] = sum(item['prob'] for item in map_list) / len(map_list)
                agg_dict['task'] = map_list[0]['task']
                agg_dict['gt'] = map_list[0]['gt']
                agg_dict['cxr_img'] = map_list[0]['cxr_img']
                agg_dict['cxr_dims'] = map_list[0]['cxr_dims']

            # save aggregate cam
            cam_path = save_dir_predictions/'cams'
            os.makedirs(cam_path, exist_ok=True)
            file_path = cam_path/file
            file = open(file_path, 'wb')
            pickle.dump(agg_dict, file)
            
    return save_dir_predictions
            

def test(args):
    logger = WandbLogger(name=args.logger_args.wandb_run_name,
                         project=args.logger_args.wandb_project_name)
    
    # get predictions and cams
    if args.model_args.config_path is not None: # ensemble model
        file = open(args.model_args.config_path)
        config = json.load(file)
        
        # map unique ckpts to tasks
        ckpt_path2tasks = {}
        for task, info in config["task2models"].items():
            for ckpt in info:
                ckpt_path = ckpt["ckpt_path"]
                if ckpt_path in ckpt_path2tasks:
                    ckpt_path2tasks[ckpt_path].append(task)
                else:
                    ckpt_path2tasks[ckpt_path] = [task]

        # save predictions and cams for each ckpt
        save_dir_all_ckpts = []
        for i, (ckpt_path, tasks) in enumerate(ckpt_path2tasks.items()):
            print(f"Running checkpoint {i+1} of {len(ckpt_path2tasks)}.")
            args.model_args.ckpt_path = ckpt_path
            save_dir = test_single_model(args, logger, tasks)
            save_dir_all_ckpts.append(save_dir)
            
        # aggregate predictions and cams
        save_dir_predictions = aggregate_ckpts(args, save_dir_all_ckpts, config)
        
    else: # single model
        save_dir_predictions = test_single_model(args, logger, CHEXPERT_TASKS)

    # save cams
    # if args.logger_args.save_cams:
    #     post_process_cams(args, save_dir_predictions)
    
    # delete cams from individual checkpoints that make up ensemble
    if args.model_args.config_path is not None and args.logger_args.delete_ckpt_cams:
        for ckpt_path in save_dir_all_ckpts:
            shutil.rmtree(ckpt_path)
    
        
if __name__ == "__main__":
    parser = TestArgParser()
    test(parser.parse_args())
