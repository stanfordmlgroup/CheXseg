import torch
import torch.nn.functional as F
from constants import *


class GradCAM(object):
    """Calculate unnormalized GradCAM saliency map.
    
    Args:
        model_dict (dict): dictionary that contains keys 'type', 'arch', and 'layer_name'.
            e.g.: model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5')
            type (str): model type (e.g. 'densenet', 'vgg', 'resnet', 'alexnet', or 'squeezenet')
            arch (obj): model architecture
            layer_name (str): name of the last convolutional layer off of which to calculate GradCAM
    
    GradCAM paper: R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra (2019)
                   “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,”
                   arXiv preprint arXiv:1610.02391v4, https://arxiv.org/pdf/1610.02391.pdf
    
    Adapted from: https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py
    """
    def __init__(self, model_dict):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        # Find the last convolutional layers based on model architecture.
        target_layer = find_target_layer(self.model_arch, layer_name)

        # Attach forward_hook and backward_hook to target_layer to capture its activations and gradients
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx):
        """Perform forward and backward pass to calculate GradCAM saliency map.
        
        Args:
            input (torch.tensor): input image tensor with shape (1, C, H, W)
            class_idx (int): class index for calculating GradCAM
        
        Return:
            saliency map (torch.tensor): unnormalized saliency map with shape (1, 1, H, W)
            logit (torch.tensor): model output
            prob (float): predicted probability for class_idx
        """
        b, c, h, w = input.size()
        if b != 1: raise Exception(f'Batch size must equal 1, but the value of b was {b}.')

        with torch.set_grad_enabled(True):
            logit = self.model_arch(input) # forward pass
        
            if self.model_arch.model.classifier.out_features == 42: # if model uses uncertainty label
                class_idx_3class = 2 + class_idx * 3 # get index of positive class of pathology
                score = logit[:, class_idx_3class].squeeze()
                probs = uncertain_logits_to_probs(logit)
                prob = float(probs[:, class_idx])
            else:
                score = logit[:, class_idx].squeeze()
                prob = float(torch.sigmoid(score))

            self.model_arch.zero_grad() # clear previous gradients
            score.backward(retain_graph=False) # compute gradients via backprop
            gradients = self.gradients['value'] # gradient of score w.r.t feature map activations (dY_c/dA)
            activations = self.activations['value'] # feature map activations (A)
            b, k, u, v = gradients.size()

            # GAP of gradients over width and height dimensions.
            # Equation (1) in GradCAM paper.
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)

            # Perform weighted combination of forward activation maps, followed by ReLU.
            # Equation (2) in GradCAM paper.
            saliency_map = (weights*activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        return saliency_map, logit, prob

    def __call__(self, input, class_idx):
        return self.forward(input, class_idx)


def find_target_layer(arch, target_layer_name):
    """Find target layer for densenet to calculate GradCAM.
    
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'
            
    Return:
        target_layer: found layer. This layer will be hooked to get forward/backward pass information.
    
    Adapted from: https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/utils.py
    """
    hierarchy = target_layer_name.split('.')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]
    return target_layer
