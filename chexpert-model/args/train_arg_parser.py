"""Define class for processing training command-line arguments."""
from .base_arg_parser import BaseArgParser
import util


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Model args
        self.parser.add_argument('--model',
                                 dest='model_args.model',
                                 default='DenseNet121',
                                 help='Model name.')
        self.parser.add_argument('--pretrained', dest='model_args.pretrained',
                                 type=util.str_to_bool, default=True,
                                 help='Use a pretrained network.')
        
        # Model args for Wildcat
        self.parser.add_argument('--num_maps',
                                 dest='model_args.wildcat_num_maps',
                                 type=int, default=1,
                                 help=('Number of feature maps used to ' +
                                       'learn class-related modalities ' +
                                       'for Wildcat model.'))
        self.parser.add_argument('--kmax',
                                 dest='model_args.wildcat_kmax',
                                 type=int, default=1,
                                 help=('Number of regions to select ' +
                                       'with highest activations for ' +
                                       'spatial pooling in Wildcat model.'))
        self.parser.add_argument('--kmin',
                                 dest='model_args.wildcat_kmin',
                                 type=int, default=None,
                                 help=('Number of regions to select ' +
                                       'with lowest activations for ' +
                                       'spatial pooling in Wildcat model.'))
        self.parser.add_argument('--alpha',
                                 dest='model_args.wildcat_alpha',
                                 type=float, default=1.0,
                                 help=('Weighting for maximum and ' +
                                       'minimum scoring regions for ' +
                                       'spatial pooling in Wildcat model.'))

        # Logger args
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, default='default',
                                 help='Experiment name.')
        self.parser.add_argument('--train_custom_csv',
                                 dest='data_args.csv',
                                 type=str, default=None,
                                 help='csv for custom dataset.')
        self.parser.add_argument('--val_custom_csv',
                                 dest='data_args.csv_dev',
                                 type=str, default=None,
                                 help='csv for custom dev dataset.')
        self.parser.add_argument('--save_top_k',
                                 dest='logger_args.save_top_k',
                                 type=int, default=10,
                                 help=('Number of checkpoints to keep ' +
                                       'before overwriting old ones.'))

        # Training args
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=50,
                                 help=('Number of epochs to train. If 0, ' +
                                       'train forever.'))
        self.parser.add_argument('--metric_name',
                                 dest='optim_args.metric_name',
                                 choices=('val_loss',
                                          'chexpert_competition_AUROC'),
                                 default='chexpert_competition_AUROC',
                                 help=('Validation metric to optimize.'))
        self.parser.add_argument('--val_check_interval',
                                 dest='optim_args.val_check_interval',
                                 type=float, default=0.1,
                                 help=('How often within one epoch to check the val set.'))
        self.parser.add_argument('--limit_train_batches',
                                 dest='optim_args.limit_train_batches',
                                 type=float, default=1.0,
                                 help=('How much of training dataset to check.'))

        # Optimizer
        self.parser.add_argument('--optimizer',
                                 dest='optim_args.optimizer',
                                 type=str, default='adam',
                                 choices=('sgd', 'adam', 'ranger'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum',
                                 dest='optim_args.sgd_momentum',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening',
                                 dest='optim_args.sgd_dampening',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--weight_decay',
                                 dest='optim_args.weight_decay',
                                 type=float, default=0.0,
                                 help='Weight decay (L2 coefficient).')
        # Learning rate
        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-4,
                                 help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler',
                                 dest='optim_args.lr_scheduler',
                                 type=str, default=None,
                                 choices=(None, 'step', 'multi_step',
                                          'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma',
                                 dest='optim_args.lr_decay_gamma',
                                 type=float, default=0.1,
                                 help=('Multiply learning rate by this ' +
                                       'value every LR step (step and ' +
                                       'multi_step only).'))
        self.parser.add_argument('--lr_decay_step',
                                 dest='optim_args.lr_decay_step',
                                 type=int, default=100,
                                 help=('Number of epochs between each ' +
                                       'multiply-by-gamma step.'))
        self.parser.add_argument('--lr_milestones',
                                 dest='optim_args.lr_milestones',
                                 type=str, default='50,125,250',
                                 help=('Epochs to step the LR when using ' +
                                       'multi_step LR scheduler.'))
        self.parser.add_argument('--lr_patience',
                                 dest='optim_args.lr_patience',
                                 type=int, default=2,
                                 help=('Number of stagnant epochs before ' +
                                       'stepping LR.'))
        # Loss function
        self.parser.add_argument('--loss_fn',
                                 dest='optim_args.loss_fn',
                                 choices=('cross_entropy',),
                                 default='cross_entropy',
                                 help='loss function.')

        # Transform arguments
        self.parser.add_argument('--scale',
                                 dest='transform_args.scale',
                                 default=320, type=int)
        self.parser.add_argument('--crop',
                                 dest='transform_args.crop',
                                 type=int, default=320)
        self.parser.add_argument('--normalization',
                                 dest='transform_args.normalization',
                                 default='imagenet',
                                 choices=('imagenet', 'chexpert_norm'))
        self.parser.add_argument('--maintain_ratio',
                                 dest='transform_args.maintain_ratio',
                                 type=util.str_to_bool, default=True)

        # Data augmentation
        self.parser.add_argument('--rotate_min',
                                 dest='transform_args.rotate_min',
                                 type=float, default=0)
        self.parser.add_argument('--rotate_max',
                                 dest='transform_args.rotate_max',
                                 type=float, default=0)
        self.parser.add_argument('--rotate_prob',
                                 dest='transform_args.rotate_prob',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_min',
                                 dest='transform_args.contrast_min',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_max',
                                 dest='transform_args.contrast_max',
                                 type=float, default=0)
        self.parser.add_argument('--contrast_prob',
                                 dest='transform_args.contrast_prob',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_min',
                                 dest='transform_args.brightness_min',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_max',
                                 dest='transform_args.brightness_max',
                                 type=float, default=0)
        self.parser.add_argument('--brightness_prob',
                                 dest='transform_args.brightness_prob',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_min',
                                 dest='transform_args.sharpness_min',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_max',
                                 dest='transform_args.sharpness_max',
                                 type=float, default=0)
        self.parser.add_argument('--sharpness_prob',
                                 dest='transform_args.sharpness_prob',
                                 type=float, default=0)
        self.parser.add_argument('--horizontal_flip_prob',
                                 dest='transform_args.horizontal_flip_prob',
                                 type=float, default=0)
