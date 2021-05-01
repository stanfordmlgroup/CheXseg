"""Select models for an ensemble and assemble the corresponding JSON config.

Usage:
    Say [search_dir] is a directory containing multiple experiments,
    then to generate a config for an ensemble:
        
        python select_ensemble.py --search_dir [search_dir]
                                  --tasks "Atelectasis,Pleural Effusion"
                                  --ckpt_pattern "*.ckpt"
                                  --max_ckpts 10
                                  --config_name "final.json"
    
    To generate a config for all tasks, do not specify the --tasks arg.
    Configs are saved to [search_dir], under the default filename "final.json".
"""


import glob
import json
import os
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pytorch_lightning import Trainer
from sklearn.metrics import roc_auc_score

import util
from constants import CHEXPERT_COMPETITION_TASKS
from data import get_loader
from statistics import mean
from train import Model


def find_checkpoints(search_dir, ckpt_pattern):
    """Recursively search search_dir, and find all ckpts matching the pattern.

    When searching, the script will skip over checkpoints for which a
    corresponding args.json does not exist. It will also ensure that all
    models were validated on the same validation set.

    Args:
        search_dir (Path): the directory in which to search
        ckpt_pattern (str): the filename pattern to match

    Returns:
        ckpts (list): list of (Path, dict) corresponding to checkpoint paths
            and the corresponding args
        csv_dev_path (str): path specifying the validation set

    """
    # Temporarily switch to search_dir to make searching easier
    cwd = os.getcwd()
    os.chdir(search_dir)
    ckpts = []

    # Recursively search for all files matching pattern
    for filename in glob.iglob('**/%s' % ckpt_pattern, recursive=True):
        ckpt_path = search_dir / filename
        ckpt_dir = ckpt_path.parents[1]
        args_path = ckpt_dir / 'args.json'
        if not args_path.exists():
            print('args.json not found for %s! Skipping.' % str(ckpt_path))
            continue

        with open(args_path) as f:
            ckpt_args = json.load(f)

        # FIXME: Make sure all validation sets are the same.

        ckpts.append((ckpt_path, ckpt_args))

    # Switch back to original working directory
    os.chdir(cwd)
    print('Found %d checkpoint(s).' % len(ckpts))
    return ckpts


def run_model(ckpt_path, ckpt_args):
    """Run a model with the specified args and output predictions.

    Args:
        ckpt_path (Path): path specifying the checkpoint
        ckpt_args (dict): args associated with the corresponding run

    Returns:
        pred_df (pandas.DataFrame): model predictions
        gt_df (pandas.DataFrame): corresponding ground-truth labels

    """
    model = Model.load_from_checkpoint(ckpt_path)

    logger_args = vars(model.logger_args)
    logger_args['save_cams'] = False
    logger_args['save_predictions'] = False
    data_args = vars(model.data_args)
    data_args['phase'] = 'valid'

    trainer = Trainer()
    trainer.test(model)
    y_pred = model.y_pred
    y_test = model.y_test
    
    tasks = model.model_args.tasks
    pred_df = pd.DataFrame({task: y_pred[:, i]
                            for i, task in enumerate(tasks)})
    gt_df = pd.DataFrame({task: y_test[:, i]
                          for i, task in enumerate(tasks)})
    return pred_df, gt_df


def get_auc_metric(task):
    """Get a metric that calculates AUC for a specified task.

    Args:
        task (str): the column over which to calculate AUC

    Returns:
        metric (function): metric operating on (pred_df, gt_df)
                           to calculate AUC for the specified task

    """
    def metric(pred_df, gt_df):
        # AUC score requires at least 1 of each class label
        if len(set(gt_df[task])) < 2:
            return None
        return roc_auc_score(gt_df[task], pred_df[task])
    return metric


def rank_models(ckpt_path2dfs, metric, maximize_metric):
    """Rank models according to the specified metric.

    Args:
        ckpt_path2dfs (dict): mapping from ckpt_path (str) to (pred_df, gt_df)
        tasks (list): list of tasks on which to evaluate checkpoints
        maximize_metric (bool): whether higher values of the metric are better
                                (as opposed to lower values)

    Returns:
        ranking (list): list containing (Path, float), corresponding to
                        checkpoint-metric pairs ranked from best to worst
                        by metric value

    """
    assert len(ckpt_path2dfs)
    ranking = []
    values = []
    for ckpt_path, (pred_df, gt_df) in ckpt_path2dfs.items():
        try:
            value = metric(pred_df, gt_df)
            if value is None:
                continue
            ranking.append((ckpt_path, value))
            values.append(value)
        except ValueError:
            continue

    # For deterministic results, break ties using checkpoint name.
    # We can do this since sort is stable.
    ranking.sort(key=lambda x: x[0])
    ranking.sort(key=lambda x: x[1], reverse=maximize_metric)
    if maximize_metric:
        assert ranking[0][1] == max(values)
    else:
        assert ranking[0][1] == min(values)
    return ranking


def get_config_list(ranking, ckpt_path2is_3class):
    """Assemble a model list for a specific task based on the ranking.

    In addition to bundling information about the ckpt_path and whether to
    model_uncertainty, the config_list also lists the value of the metric to
    aid debugging.

    Args:
        ranking (list): list containing (Path, float), corresponding to
                        checkpoint-metric pairs ranked from best to worst
                        by metric value
        ckpt_path2is_3class (dict): mapping from ckpt_path to is_3class
                                    (whether to model_uncertainty)

    Returns:
        config_list (list): list bundling information about ckpt_path,
                            model_uncertainty, and metric value

    """
    config_list = []
    for ckpt_path, value in ranking:
        is3_class = ckpt_path2is_3class[ckpt_path]
        ckpt_info = {'ckpt_path': str(ckpt_path),
                     'is_3class': is3_class,
                     'value': value}
        config_list.append(ckpt_info)
    return config_list


def assemble_config(aggregation_method, task2models):
    """Assemble the entire config for dumping to JSON.
    
    Args:
        aggregation_method (str): the aggregation method to use
                                  during ensemble prediction
        task2models (dict): mapping from task to the associated
                            config_list of models
    
    Returns:
        (dict): dictionary representation of the ensemble config,
                ready for dumping to JSON
    """
    return {'aggregation_method': aggregation_method,
            'task2models': task2models}


def parse_script_args():
    """Parse command line arguments.

    Returns:
        args (Namespace): parsed command line arguments

    """
    parser = ArgumentParser()

    parser.add_argument('--search_dir',
                        type=str,
                        required=True,
                        help='Directory in which to search for checkpoints')

    parser.add_argument('--ckpt_pattern',
                        type=str,
                        default='iter_*.pth.tar',
                        help='Pattern for matching checkpoint files')

    parser.add_argument('--max_ckpts',
                        type=int,
                        default=10,
                        help='Max. number of checkpoints to select')

    parser.add_argument('--tasks',
                        type=str,
                        help='Prediction tasks used to rank ckpts')
    
    parser.add_argument('--aggregation_method',
                        type=str,
                        default='mean',
                        help='Aggregation method to specify in config')

    parser.add_argument('--config_name',
                        type=str,
                        default='final.json',
                        help='Name for output JSON config')

    args = parser.parse_args()

    # If no task is specified, build config for CheXpert competition tasks
    if args.tasks is None:
        args.tasks = CHEXPERT_COMPETITION_TASKS
    else:
        args.tasks = util.args_to_list(args.tasks, allow_empty=True,
                                       arg_type=str)
    return args


if __name__ == '__main__':
    args = parse_script_args()
    search_dir = Path(args.search_dir)

    # Retrieve all checkpoints that match the given pattern
    ckpts = find_checkpoints(search_dir, args.ckpt_pattern)

    # Get predictions for all checkpoints that were found
    ckpt_path2dfs = {}
    ckpt_path2is_3class = {}
    for i, (ckpt_path, ckpt_args) in enumerate(ckpts):
        print('Evaluating checkpoint (%d/%d).' % (i + 1, len(ckpts)))
        pred_df, gt_df = run_model(ckpt_path, ckpt_args)
        ckpt_path2dfs[ckpt_path] = (pred_df, gt_df)
        is_3class = ckpt_args['model_args']['model_uncertainty']
        ckpt_path2is_3class[ckpt_path] = is_3class

    # Rank the checkpoints for each task
    task2models = {}
    for task in args.tasks:
        print(f'Ranking checkpoints for {task}.')
        metric = get_auc_metric(task)
        ranking = rank_models(ckpt_path2dfs, metric, maximize_metric=True)
        ranking = ranking[:min(args.max_ckpts, len(ranking))]
        task2models[task] = get_config_list(ranking, ckpt_path2is_3class)

    # Assemble and write the ensemble config file
    print('Writing config file to %s.' % str(search_dir / args.config_name))
    config = assemble_config(args.aggregation_method, task2models)
    with open(search_dir / args.config_name, 'w+') as f:
        json.dump(config, f, indent=4)
