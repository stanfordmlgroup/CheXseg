import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from captum.attr import NoiseTunnel
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

    
def integrated_gradient(img, model, model_args, task, n_steps= N_STEPS_IG, 
                                    internal_batch_size = INTERNAL_BATCH_SIZE_IG, noise_tunnel_flag = False, 
                                            stdevs = DEFAULT_STDEV_NT, n_samples = N_SAMPLES_NT):
    
    model_infer = inference_model(model, task, model_args.model_uncertainty)
    logits = model(img).detach().cpu()
    probs = model_infer(img).detach().cpu().item()
    
    integrated_gradients = IntegratedGradients(model_infer)
    
    with torch.set_grad_enabled(True):
        if(not noise_tunnel_flag):

            attributions_ig = integrated_gradients.attribute(img, n_steps = n_steps, 
                                                             internal_batch_size = internal_batch_size)
            attribution =  attributions_ig

        else:

            noise_tunnel = NoiseTunnel(integrated_gradients)
            attributions_ig_nt = noise_tunnel.attribute(img, n_samples=n_samples, 
                                              nt_type='smoothgrad_sq',stdevs=stdevs, n_steps = n_steps, 
                                                                    internal_batch_size = internal_batch_size)

            attribution = attributions_ig_nt

    attribution = attribution[0].detach().cpu()
    attribution = attribution.mean(axis=0)
    attribution = attribution.unsqueeze(0).unsqueeze(0)
    
    return attribution, logits, probs
    
    
