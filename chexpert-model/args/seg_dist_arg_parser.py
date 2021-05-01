"""Define class for processing training command-line arguments."""
# from .seg_base_arg_parser import SegBaseArgParser
# import util
import argparse
import copy
import json
from pathlib import Path

import util
from constants import *


class SegDistArgParser(object):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CXR')

        # Logger args
        self.parser.add_argument('--save_dir',
                                 dest='logger_args.save_dir',
                                 type=str, default='./logs',
                                 help='Directory to save model data.')
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, default='default',
                                 help='Experiment name.')
        self.parser.add_argument('--wandb_project_name',
                                 dest='logger_args.wandb_project_name',
                                 type=str, default=None,
                                 help='Project name for W&B tracking.')
        self.parser.add_argument('--wandb_run_name',
                                 dest='logger_args.wandb_run_name',
                                 type=str, default=None,
                                 help='Run name for W&B tracking.')
        self.parser.add_argument('--num_valid_per_epoch',
                                 dest='logger_args.num_valid_per_epoch',
                                 type=int, default=1,
                                 help='Number of times to validate per training epoch.')

        # Data args                              
        self.parser.add_argument('--test_set',
                                 dest='data_args.test_set',
                                 type=str, default='test',
                                 help='Set of data to test on')
        self.parser.add_argument('--masks_path',
                                 dest='data_args.masks_path',
                                 type=str, default=None,
                                 help='Path of masks path to read data from')       
        self.parser.add_argument('--images_path',
                                 dest='data_args.images_path',
                                 type=str, default=None,
                                 help='Path of images to use for transfer dataset')                    

        # Model args
        self.parser.add_argument('--config_path',
                                 dest='model_args.config_path',
                                 type=str, default=None,
                                 help='Path to train checkpoint and hyperparams')
        self.parser.add_argument('--seed',
                                 dest='model_args.seed',
                                 type=int, default=17,
                                 help='Seed for random.')

        # Optim args
        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-4,
                                 help='Initial learning rate.')
        self.parser.add_argument('--temperature',
                                 dest='optim_args.temperature',
                                 type=float, default=10.,
                                 help='BCE sigmoid temperature.')

        # Training args
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=100,
                                 help=('Number of epochs to train.'))

        # Transform arguments
        self.parser.add_argument('--scale',
                                 dest='transform_args.scale',
                                 default=320, type=int)

    def fix_nested_namespaces(self, args):
        """Makes sure that nested namespaces work
            Args:
                args: argsparse.Namespace object containing all the arguments
            e.g args.data_args.batch_size

            Obs: Only one level of nesting is supported.
        """
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    
    def parse_args(self):
        """Parse command-line arguments and set up directories and other run
        args for training and testing."""
        args = self.parser.parse_args()

        # Make args a nested Namespace
        self.fix_nested_namespaces(args)

        save_dir = Path(args.logger_args.save_dir) /\
                args.logger_args.experiment_name
        args_save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save args to a JSON file in the args save directory.
        args_path = args_save_dir / 'args.json'
        with open(args_path, 'w') as fh:
            args_dict = self.namespace_to_dict(args)
            json.dump(args_dict, fh, indent=4,
                      sort_keys=True)
            fh.write('\n')

        args.logger_args.save_dir = save_dir

        return args

    def namespace_to_dict(self, args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = self.namespace_to_dict(obj)

        return args_dict