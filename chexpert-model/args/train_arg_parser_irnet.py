from .train_arg_parser import TrainArgParser
import util

class TrainArgParserIRNet(TrainArgParser):
    def __init__(self):
        super(TrainArgParserIRNet, self).__init__()
        self.is_training = True

        # Args for IRNET
        self.parser.add_argument('--irn_crop_size',
                                 dest='irnet_args.crop_size',
                                 type=int,
                                 default='320',
                                 help='Crop Size')
        
        self.parser.add_argument('--irn_network',
                                 dest='irnet_args.network',
                                 type=str,
                                 default='models.resnet_50_irn')
        
        self.parser.add_argument('--irn_best_model_name',
                                 dest='irnet_args.best_model_name',
                                 type=str)
        
        self.parser.add_argument('--irn_pseudo_labels_save_dir',
                                 dest='irnet_args.pseudo_labels_save_dir',
                                 type=str)

        self.parser.add_argument('--irn_is_training',
                                 dest='irnet_args.is_training',
                                 type=util.str_to_bool)
        
        