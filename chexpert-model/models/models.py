import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from .wildcat_pooling import WildcatPool2d, ClassWisePool

import torch


class PretrainedModel(nn.Module):
    """Pretrained model, either from Cadene or TorchVision."""
    def __init__(self):
        super(PretrainedModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError('Subclass of PretrainedModel must implement forward.')

    def fine_tuning_parameters(self, boundary_layers, lrs):
        """Get a list of parameter groups that can be passed to an optimizer.

        Args:
            boundary_layers: List of names for the boundary layers.
            lrs: List of learning rates for each parameter group, from earlier to later layers.

        Returns:
            param_groups: List of dictionaries, one per parameter group.
        """

        def gen_params(start_layer, end_layer):
            saw_start_layer = False
            for name, param in self.named_parameters():
                if end_layer is not None and name == end_layer:
                    # Saw the last layer -> done
                    return
                if start_layer is None or name == start_layer:
                    # Saw the first layer -> Start returning layers
                    saw_start_layer = True

                if saw_start_layer:
                    yield param

        if len(lrs) != boundary_layers + 1:
            raise ValueError('Got {} param groups, but {} learning rates'.format(boundary_layers + 1, len(lrs)))

        # Fine-tune the network's layers from encoder.2 onwards
        boundary_layers = [None] + boundary_layers + [None]
        param_groups = []
        for i in range(len(boundary_layers) - 1):
            start, end = boundary_layers[i:i+2]
            param_groups.append({'params': gen_params(start, end), 'lr': lrs[i]})

        return param_groups


class TorchVisionModel(PretrainedModel):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """
    def __init__(self, model_fn, tasks, model_args):
        super(TorchVisionModel, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty

        self.model = model_fn(pretrained=model_args.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.is_resnet = type(self.model) is models.resnet.ResNet
        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)
        print("Number of tasks:", num_outputs)
        if self.is_resnet:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_outputs)
        else:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        if self.is_resnet:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)

            return x
        else:
            x = self.model.features(x)
            x = F.relu(x, inplace=True)
            x = self.pool(x).view(x.size(0), -1)
            x = self.model.classifier(x)
            return x


class WildcatDenseNet121(PretrainedModel):
    """
    Adapted from: https://github.com/durandtibo/wildcat.pytorch
    """
    def __init__(self, tasks, model_args):
        super(WildcatDenseNet121, self).__init__()

        self.tasks = tasks
        self.model_uncertainty = model_args.model_uncertainty
        self.num_maps = model_args.wildcat_num_maps
        self.kmax = model_args.wildcat_kmax
        self.kmin = model_args.wildcat_kmin
        self.alpha = model_args.wildcat_alpha

        self.model = models.densenet121(pretrained=model_args.pretrained)
        
        # Multi-map transfer layer (classification)
        if self.model_uncertainty:
            num_outputs = 3 * len(tasks)
        else:
            num_outputs = len(tasks)
        num_ftrs = self.model.classifier.in_features
        MC = self.num_maps * num_outputs
        self.model.classifier = nn.Sequential(
            nn.Conv2d(num_ftrs, MC, kernel_size=1, stride=1, padding=0, bias=True))
        
        # Class-wise pooling and Wildcat pooling
        self.spatial_pooling = nn.Sequential()
        self.spatial_pooling.add_module('class_wise', ClassWisePool(self.num_maps))
        self.spatial_pooling.add_module('spatial', WildcatPool2d(self.kmax, self.kmin, self.alpha))

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.classifier(x)
        x = self.spatial_pooling(x)
        return x


class DenseNet121(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet121, self).__init__(models.densenet121, tasks, model_args)
        self.gradcam_target_layer = 'model_features_norm5'


class DenseNet161(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet161, self).__init__(models.densenet161, tasks, model_args)
        self.gradcam_target_layer = 'model_features_norm5'


class DenseNet201(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(DenseNet201, self).__init__(models.densenet201, tasks, model_args)
        self.gradcam_target_layer = 'model.features.norm5'


class ResNet101(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet101, self).__init__(models.resnet101, tasks, model_args)
        self.gradcam_target_layer = 'model.layer4.2.conv3'

class ResNet18(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet18, self).__init__(models.resnet18, tasks, model_args)
        self.gradcam_target_layer = 'model.layer4.1.conv2'


class ResNet152(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(ResNet152, self).__init__(models.resnet152, tasks, model_args)
        self.gradcam_target_layer = 'model.layer4.2.conv3'


class Inceptionv3(TorchVisionModel):
    def __init__(self, tasks, model_args):
        super(Inceptionv3, self).__init__(models.inception_v3, tasks, model_args)
        self.gradcam_target_layer = 'model.Mixed_7c.branch_pool.conv'

