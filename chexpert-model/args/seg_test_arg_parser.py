"""Define class for processing training command-line arguments."""
from .seg_base_arg_parser import SegBaseArgParser
import util


class SegTestArgParser(SegBaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(SegTestArgParser, self).__init__()
        self.is_training = False

        # Logger args
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, default='default',
                                 help='Experiment name.')
        
        self.parser.add_argument('--output_labels_save_dir',
                                 dest='logger_args.output_labels_save_dir',
                                 type=str, default='./logs',
                                 help='Experiment name.')

        # Data args                              
        self.parser.add_argument('--test_set',
                                 dest='data_args.test_set',
                                 type=str, default='test',
                                 help='Set of data to test on')
        self.parser.add_argument('--test_masks_path',
                                 dest='data_args.test_masks_path',
                                 type=str, default=None,
                                 help='Path of custom masks to test on.')

        # Model args
        self.parser.add_argument('--config_path',
                                 dest='model_args.config_path',
                                 type=str, default=None,
                                 help='Path to train checkpoint and hyperparams')
        self.parser.add_argument('--distillation_teacher_config_path',
                                dest='model_args.distillation_teacher_config_path',
                                type=str, default=None,
                                help='Path to teacher for using args')
        

        # self.parser.add_argument('--ckpt_path',
        #                          dest='model_args.ckpt_path',
        #                          type=str, default=None,
        #                          help=('Checkpoint path for tuning. ' +
        #                                'If None, start from scratch.'))