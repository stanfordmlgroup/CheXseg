from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr import NoiseTunnel
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

def uncertain_logits_to_probs(logits):
    """Convert explicit uncertainty modeling logits to probabilities P(is_abnormal).

    Args:
        logits: Input of shape (batch_size, num_tasks * 3).

    Returns:
        probs: Output of shape (batch_size, num_tasks).
            Position (i, j) interpreted as P(example i has pathology j).
    """
    b, n_times_d = logits.size()
    d = 3
    if n_times_d % d:
        raise ValueError('Expected logits dimension to be divisible by {}, got size {}.'.format(d, n_times_d))
    n = n_times_d // d

    logits = logits.view(b, n, d)
    probs = F.softmax(logits[:, :, 1:], dim=-1)
    probs = probs[:, :, 1]

    return probs


class inference_model(nn.Module):
    def __init__(self, model,task, model_uncertainty):
        super(inference_model, self).__init__()
        
        self.model = model
        self.task=task
        self.get_probs = uncertain_logits_to_probs if model_uncertainty else torch.sigmoid
    
    def forward(self,x):
        
        preds = self.model(x)

        probs = self.get_probs(preds)[0]
        idx = self.model.tasks.index(self.task)
        return probs[idx].unsqueeze(0)

    
class grad(LayerGradCam):
    
    def __init__(self,model_infer, layer_name):
        LayerGradCam.__init__(self,model_infer,layer_name)
    
    def has_convergence_delta(self):
        return False


def gradcam_nt(img, model, model_args, task, n_samples = N_SAMPLES_NT, stdevs = DEFAULT_STDEV_NT, noise_tunnel_flag = False):
    
    model_infer = inference_model(model, task,model_args.model_uncertainty)
    logits = model(img).detach().cpu()
    probs = model_infer(img).detach().cpu().item()
    
    mod = grad(model_infer,model.model.features.norm5)
    
    if(not noise_tunnel_flag):
        attributions_gc = mod.attribute(img, relu_attributions = True)
        attributions = attributions_gc
    
    else:
        noise_tunnel = NoiseTunnel(mod)
        attributions_gc_nt = noise_tunnel.attribute(img, n_samples=n_samples,
                                                nt_type='smoothgrad_sq',stdevs=stdevs, relu_attributions = True)
        attributions = attributions_gc_nt
    
    attributions = attributions.detach().cpu()
    return attributions, logits, probs

