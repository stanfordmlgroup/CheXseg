import sys
import torch
import torch.nn as nn
from torch.autograd import Function


class WildcatPool2dFunction(Function):
    """
    Wildcat paper: T. Durand, T. Mordan, N. Thome, and M. Cord (2017),
                   “WILDCAT: Weakly Supervised Learning of Deep ConvNets for
                   Image Classification, Pointwise Localization and Segmentation,”
                   IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017).
                   
    Adapted from: https://github.com/durandtibo/wildcat.pytorch
    """
    @staticmethod
    def forward(ctx, input, kmax, kmin, alpha):
        def get_positive_k(k, n):
            if k <= 0:
                return 0
            elif k < 1:
                return round(k * n)
            elif k > n:
                return int(n)
            else:
                return int(k)
        
        b, c, h, w = input.shape
        ctx.num_regions = h * w
        ctx.kmax = get_positive_k(kmax, ctx.num_regions)
        ctx.kmin = get_positive_k(kmin, ctx.num_regions)
        ctx.alpha = alpha
        
        sorted, indices = torch.sort(input.view(b, c, ctx.num_regions), dim=2, descending=True)

        # indices of the kmax regions with highest activations
        ctx.indices_max = indices.narrow(2, 0, ctx.kmax)
        # kmax regions with highest activations
        activations_max = sorted.narrow(2, 0, ctx.kmax)
        # average scores of kmax regions with highest activations - LHS of Equation (3) sum in Wildcat paper
        avg_activations_max = activations_max.sum(2).div_(ctx.kmax)
        output = avg_activations_max

        if ctx.kmin > 0 and ctx.alpha != 0:
            # indices of the kmin regions with lowest activations
            ctx.indices_min = indices.narrow(2, ctx.num_regions - ctx.kmin, ctx.kmin)
            # kmax regions with lowest activations
            activations_min = sorted.narrow(2, ctx.num_regions - ctx.kmin, ctx.kmin)
            # average scores of kmin regions with lowest activations - RHS of Equation (3) sum in Wildcat paper
            avg_activations_min = activations_min.sum(2).mul_(ctx.alpha / ctx.kmin)
            output = avg_activations_max + avg_activations_min

        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradients are backpropagated only within the k+ + k− selected regions;
        # all other gradients are discarded.
        input = ctx.saved_tensors[0]
        b, c, h, w = input.shape

        # get k+ regions of grad_output
        grad_output_max = grad_output.view(b, c, 1).expand(b, c, ctx.kmax)
        grad_input_max = grad_output.new_zeros((b, c, ctx.num_regions))
        # write values from grad_output_max (k+ regions) into grad_input
        # at indices specified in ctx.indices_max
        grad_input_max = grad_input_max.scatter_(2, ctx.indices_max, grad_output_max)
        grad_input_max = grad_input_max.div_(ctx.kmax)
        grad_input = grad_input_max

        if ctx.kmin > 0 and ctx.alpha != 0:
            # get k- regions of grad_output
            grad_output_min = grad_output.view(b, c, 1).expand(b, c, ctx.kmin)
            grad_input_min = grad_output.new_zeros((b, c, ctx.num_regions))
            grad_input_min = grad_input_min.scatter_(2, ctx.indices_min, grad_output_min)
            grad_input_min = grad_input_min.mul_(ctx.alpha / ctx.kmin)
            grad_input = grad_input_max + grad_input_min

        return grad_input.view(b, c, h, w), None, None, None


class WildcatPool2d(nn.Module):
    """Adapted from: https://github.com/durandtibo/wildcat.pytorch"""
    def __init__(self, kmax, kmin, alpha):
        super(WildcatPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha

    def forward(self, input):
        return WildcatPool2dFunction.apply(input, self.kmax, self.kmin, self.alpha)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax) + ', kmin=' + str(self.kmin) + ', alpha=' + str(
            self.alpha) + ')'


class ClassWisePoolFunction(Function):
    """Adapted from: https://github.com/durandtibo/wildcat.pytorch"""
    @staticmethod
    def forward(ctx, input, num_maps):
        b, c, h, w = input.shape
        ctx.num_maps = num_maps

        if c % ctx.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels ' +
                  'has to be a multiple of the number of maps per class.')
            sys.exit(-1)

        num_outputs = int(c / ctx.num_maps)
        x = input.view(b, num_outputs, ctx.num_maps, h, w)
        output = torch.sum(x, 2) / ctx.num_maps
        
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        b, c, h, w = input.shape

        num_outputs = grad_output.shape[1]
        grad_input = grad_output.view(b, num_outputs, 1, h, w)
        grad_input = grad_input.expand(b, num_outputs, ctx.num_maps, h, w).contiguous()
        grad_input = grad_input.view(b, c, h, w)

        return grad_input, None


class ClassWisePool(nn.Module):
    """Adapted from: https://github.com/durandtibo/wildcat.pytorch"""
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(num_maps=self.num_maps)
