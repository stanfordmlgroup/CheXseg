"""Define class for processing testing command-line arguments."""
import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--inference_only',
                                 action='store_true',
                                 help=('If set, then only do inference. Useful'+
                                       ' when the csv has uncertainty label'))
        # Data args
        self.parser.add_argument('--phase',
                                 dest='data_args.phase',
                                 type=str, default='valid',
                                 choices=('train', 'valid', 'test'))
        self.parser.add_argument('--test_groundtruth',
                                 dest='data_args.gt_csv',
                                 type=str, default=None,
                                 help=('csv file if custom dataset'))
        self.parser.add_argument('--test_image_paths',
                                 dest='data_args.paths_csv',
                                 type=str, default=None,
                                 help=('csv file if custom dataset'))
        self.parser.add_argument('--together',
                                 dest='data_args.together',
                                 type=str, default=True,
                                 help=('whether we have integrated test csv'))
        self.parser.add_argument('--test_csv',
                                 dest='data_args.test_csv',
                                 type=str, default=None,
                                 help=('csv file for integrated test set'))
        # Logger args
        self.parser.add_argument('--save_cams',
                                 dest='logger_args.save_cams',
                                 type=util.str_to_bool, default=False,
                                 help=('If true, will save cams to ' +
                                       'experiment_folder/cams'))

        self.parser.add_argument('--save_train_cams',
                                 dest='logger_args.save_train_cams',
                                 type=util.str_to_bool, default=False,
                                 help=('If true, will consider the train dataset and save ' +
                                       'the train cams'))
                                       
        self.parser.add_argument('--delete_ckpt_cams',
                                 dest='logger_args.delete_ckpt_cams',
                                 type=util.str_to_bool, default=True,
                                 help=('If true, will delete cams from ' +
                                       'individual checkpoints and save ' +
                                       'only ensemble cams.'))
        self.parser.add_argument('--save_cams_wandb',
                                 dest='logger_args.save_cams_wandb',
                                 type=util.str_to_bool, default=False,
                                 help=('If true, will save cams to w&b'))
        self.parser.add_argument('--only_evaluation_cams',
                                 dest='logger_args.only_evaluation_cams',
                                 type=util.str_to_bool, default=True,
                                 help=('If true, will only generate cams ' +
                                       'on evaluation labels. Only ' +
                                       'relevant if --save_cams is True'))
        self.parser.add_argument('--only_competition_cams',
                                 dest='logger_args.only_competition_cams',
                                 type=util.str_to_bool, default=False,
                                 help='Whether to only output cams for' +
                                 'competition categories.')
        self.parser.add_argument('--aggregation_method',
                                 dest='logger_args.aggregation_method',
                                 type=str, default= 'mean',
                                 help='Aggregation method for preds and cams')
        
        # Cam args
        self.parser.add_argument('--save_dir_predictions',
                                 dest='logger_args.save_dir_predictions',
                                 type=str, default=None,
                                 help='Directory for saving predictions and cams.')
        
        self.parser.add_argument('--cam_method',
                                 dest='logger_args.cam_method',
                                 type=str, default='gradcam',
                                 help='Name of the localization method')
        
        self.parser.add_argument('--noise_tunnel',
                                 dest='logger_args.noise_tunnel',
                                 type=util.str_to_bool, default=False,
                                 help='If to use noise tunnel for cams')
        
        self.parser.add_argument('--add_noise',
                                 dest='logger_args.add_noise',
                                 type= float, default=0.1,
                                 help='stdev of noise in noise_tunnel')
        
        # Model args
        self.parser.add_argument('--config_path',
                                 dest='model_args.config_path',
                                 type=str, default=None)
        self.parser.add_argument('--calibrate',
                                 dest='model_args.calibrate',
                                 type=util.str_to_bool, default=False,
                                 help='Compute calibrated probabilities.')
