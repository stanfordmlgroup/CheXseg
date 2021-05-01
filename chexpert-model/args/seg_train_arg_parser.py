"""Define class for processing training command-line arguments."""
from .seg_base_arg_parser import SegBaseArgParser
import util


class SegTrainArgParser(SegBaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(SegTrainArgParser, self).__init__()
        self.is_training = True

        # Logger args
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, default='default',
                                 help='Experiment name.')

        # Model args
        self.parser.add_argument('--architecture',
                                 dest='model_args.architecture',
                                 type=str, default='DeepLabV3Plus',
                                 help=('Decoder archetecture.'))
        self.parser.add_argument('--encoder',
                                 dest='model_args.encoder',
                                 default='se_resnext50_32x4d',
                                 help='Segmentation encoder.')
        self.parser.add_argument('--encoder_weights',
                                 dest='model_args.encoder_weights',
                                 default='imagenet',
                                 help='Segmentation encoder weights.')
        self.parser.add_argument('--encoder_weights_type',
                                 dest='model_args.encoder_weights_type',
                                 default='',
                                 help='Segmentation encoder weights type .')
        self.parser.add_argument('--seed',
                                 dest='model_args.seed',
                                 type=int, default=17,
                                 help='Seed for random.')
                                 
        # Learning rate
        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-4,
                                 help='Initial learning rate.')
        
        # Save validation results on common pathologies
        self.parser.add_argument('--valid_common_pathologies',
                                 dest='optim_args.valid_common_pathologies',
                                 type=bool, default=False,
                                 help='Save best checkpoints on performance of most common pathologies.')

        # Data args                              
        self.parser.add_argument('--train_set',
                                 dest='data_args.train_set',
                                 type=str, default='train',
                                 help='Set of data to train on')
        self.parser.add_argument('--valid_set',
                                 dest='data_args.valid_set',
                                 type=str, default='valid',
                                 help='Set of data to validate on')
        self.parser.add_argument('--train_masks_path',
                                 dest='data_args.train_masks_path',
                                 type=str, default=None,
                                 help='Path of custom masks to train on.')
        self.parser.add_argument('--semi_supervised',
                                 dest='data_args.semi_supervised',
                                 type=bool, default=False,
                                 help='Train on strongly-labeled validation data.')
        self.parser.add_argument('--ss_expert_annotations_masks_path',
                                 dest='data_args.ss_expert_annotations_masks_path',
                                 type=str, default=None,
                                 help='Path of masks labeled by radiologists (for ss).')
        self.parser.add_argument('--ss_dnn_generated_masks_path',
                                 dest='data_args.ss_dnn_generated_masks_path',
                                 type=str, default=None,
                                 help='Path of masks created by CAM or IRNet (for ss).')
        self.parser.add_argument('--task',
                                 dest='data_args.task',
                                 type=str, default=None,
                                 help='Train or test on a single task.')   
        self.parser.add_argument('--weighted',
                                 dest='data_args.weighted',
                                 type=bool, default=False,
                                 help='Weight strong labels more than weak')
        self.parser.add_argument('--strong_labels_weight',
                                 dest='data_args.strong_labels_weight',
                                 type=float, default=0.5,
                                 help='Weight for strong labels')                              
        
        # Training args
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=40,
                                 help=('Number of epochs to train. If 0, ' +
                                       'train forever.'))

