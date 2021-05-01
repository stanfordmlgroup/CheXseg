"""Define base class for processing command-line arguments."""
import argparse
import copy
import json
from pathlib import Path

import util
from constants import *


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CXR')

        # Logger args
        self.parser.add_argument('--save_dir',
                                 dest='logger_args.save_dir',
                                 type=str, default='./logs',
                                 help='Directory to save model data.')
        self.parser.add_argument('--wandb_project_name',
                                 dest='logger_args.wandb_project_name',
                                 type=str, default=None,
                                 help='Project name for W&B tracking.')
        self.parser.add_argument('--wandb_run_name',
                                 dest='logger_args.wandb_run_name',
                                 type=str, default=None,
                                 help='Run name for W&B tracking.')

        # Data args
        self.parser.add_argument('--batch_size',
                                 dest='data_args.batch_size',
                                 type=int, default=16,
                                 help='Batch size for training / evaluation.')
        self.parser.add_argument('--toy',
                                 dest='data_args.toy',
                                 type=util.str_to_bool, default=False,
                                 help='Use toy dataset.')
        self.parser.add_argument('--dataset',
                                 dest='data_args.dataset',
                                 type=str, default='chexpert',
                                 help=('Name of dataset. Directories ' +
                                       'are specified in constants.'))
        self.parser.add_argument('--num_workers',
                                 dest='data_args.num_workers',
                                 type=int, default=16,
                                 help='Number of threads for the DataLoader.')
        # TODO: rename this to something more general
        self.parser.add_argument('--uncertain_map_path',
                                 dest='data_args.uncertain_map_path',
                                 type=str, default=None,
                                 help=('Path to CSV file which will ' +
                                       'replace the training CSV.'))
        # Model args
        self.parser.add_argument('--ckpt_path',
                                 dest='model_args.ckpt_path',
                                 type=str, default=None,
                                 help=('Checkpoint path for tuning. ' +
                                       'If None, start from scratch.'))
        # TODO: read this from saved model
        self.parser.add_argument('--model_uncertainty',
                                 dest='model_args.model_uncertainty',
                                 type=util.str_to_bool, default=False,
                                 help=('If true, model uncertainty ' +
                                       'explicitly with (+, -, u) outputs.'))
        # Run args
        self.parser.add_argument('--gpu_ids',
                                 type=str, default='-1',
                                 help=('Comma-separated list of GPU IDs. ' +
                                       'Default -1 uses all available GPUs.'))

        self.is_training = None

    def namespace_to_dict(self, args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = self.namespace_to_dict(obj)

        return args_dict

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

        if self.is_training:
            log_name = "train_log.txt"
            # Set up training tasks for the model to output and train on.
            # tasks = CHEXPERT_SEGMENTATION_CLASSES
            # args.model_args.__dict__[TASKS] = tasks
            tasks = DATASET2TASKS[args.data_args.dataset]
            args.model_args.__dict__[TASKS] = tasks

            # Set up model save directory for logging.
            save_dir = Path(args.logger_args.save_dir) /\
                args.logger_args.experiment_name
            args_save_dir = save_dir

        else:
            if ((args.model_args.config_path is None)
                    and (args.model_args.ckpt_path is None)):
                raise ArgumentError("Must pass in a configuration file or " +
                                    "ckpt path during testing.")

            if ((args.model_args.config_path is not None)
                    and (args.model_args.ckpt_path is not None)):
                print("Provided config path and ckpt path. Using config path.")
                args.model_args.ckpt_path = None

            log_name = f"{args.data_args.phase}_log.txt"

            if args.model_args.config_path is not None:
                # Obtain save dir from config path name.
                config_path = Path(args.model_args.config_path)
                save_dir = Path(args.logger_args.save_dir) / config_path.stem
                args.logger_args.experiment_name = config_path.stem

            else:
                # Obtain save dir from ckpt path.
                save_dir = Path(args.model_args.ckpt_path).parent
                args.logger_args.experiment_name = save_dir.name

            # Make directory to save results.
            results_dir = save_dir / "results" / args.data_args.phase
            results_dir.mkdir(parents=True, exist_ok=True)

            args_save_dir = results_dir

        # Create the model save directory.
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save args to a JSON file in the args save directory.
        args_path = args_save_dir / 'args.json'
        with open(args_path, 'w') as fh:
            args_dict = self.namespace_to_dict(args)
            json.dump(args_dict, fh, indent=4,
                      sort_keys=True)
            fh.write('\n')

        args.logger_args.save_dir = save_dir
        args.logger_args.log_path = args.logger_args.save_dir / log_name

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training

        # Set up output dir (test mode only)
        if not self.is_training:
            args.logger_args.results_dir = results_dir

        return args
