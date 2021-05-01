"""Define base class for processing command-line arguments."""
import argparse
import copy
import json
from pathlib import Path

import util
from constants import *


class SegBaseArgParser(object):
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
                                 type=int, default=8,
                                 help='Batch size for training / evaluation.')
        self.parser.add_argument('--num_workers',
                                 dest='data_args.num_workers',
                                 type=int, default=12,
                                 help='Number of threads for the DataLoader.')
        self.parser.add_argument('--eval_masks_path',
                                 dest='data_args.eval_masks_path',
                                 type=str, default=None,
                                 help='Path of custom masks for evaluation.')

        # Model argss
        self.parser.add_argument('--device',
                                 dest='model_args.device',
                                 default='cuda',
                                 help='Device.')

        # Run args
        self.parser.add_argument('--gpu_ids',
                                 type=str, default='-1',
                                 help=('Comma-separated list of GPU IDs. ' +
                                       'Default -1 uses all available GPUs.'))


        # Transform arguments
        self.parser.add_argument('--scale',
                                 dest='transform_args.scale',
                                 default=320, type=int)

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

        # args.data_args.classes = CHEXPERT_SEGMENTATION_SAMPLE_CLASSES

        if self.is_training:
            log_name = "train_log.txt"
            # Set up training tasks for the model to output and train on.
            #set classes
            if args.data_args.task:
                args.data_args.classes = [args.data_args.task]
            else:
                args.data_args.classes = CHEXPERT_SEGMENTATION_CLASSES
            # Set up model save directory for logging.
            save_dir = Path(args.logger_args.save_dir) /\
                args.logger_args.experiment_name
            args_save_dir = save_dir

            # make sure custom masks path is included when doing semi-supervised
            if (args.data_args.semi_supervised):
                if (not args.data_args.ss_expert_annotations_masks_path) or (not args.data_args.ss_dnn_generated_masks_path):
                    raise ArgumentError("Must pass in two custom masks paths for semi-supervised learning.")
            else:
                if (not args.data_args.train_masks_path) or (not args.data_args.eval_masks_path):
                    raise ArgumentError("Must pass in train and evaluation masks.")
            if args.model_args.encoder_weights == 'None':
                args.model_args.encoder_weights = None
        else:
            if (args.model_args.config_path is None):
                raise ArgumentError("Must pass in a configuration file.")
            if (args.data_args.eval_masks_path is None):
                raise ArgumentError("Must pass in evaluation masks")
            log_name = f"{args.data_args.test_set}_log.txt"

            save_dir = Path(args.model_args.config_path)
            args.logger_args.experiment_name = save_dir.name

            # Make directory to save results.
            results_dir = save_dir / "results" / args.data_args.test_set
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
