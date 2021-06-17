import torch
import numpy as np

import torch.nn.functional as F
# from scipy.spatial.distance import squareform

eps = 1e-9


def dcor(x, y):
    m,_ = x.shape
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    dx = pairwise_dist(x)
    dy = pairwise_dist(y)

    dx_m = dx - dx.mean(dim=0)[None, :] - dx.mean(dim=1)[:, None] + dx.mean()
    dy_m = dy - dy.mean(dim=0)[None, :] - dy.mean(dim=1)[:, None] + dy.mean()

    dcov2_xy = (dx_m * dy_m).sum()/float(m * m) 
    dcov2_xx = (dx_m * dx_m).sum()/float(m * m) 
    dcov2_yy = (dy_m * dy_m).sum()/float(m * m) 

    dcor = torch.sqrt(dcov2_xy)/torch.sqrt((torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy)).clamp(min=0) + eps)

    return dcor

def pairwise_distances(x, y=None):
    """
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between
        x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        source:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return dist

def pairwise_dist(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2, -1).reshape((-1,1))
    output = -2*torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    return torch.sqrt(output.clamp(min=0) + eps)



def mask_softmax(x, lengths):#, dim=1)
    mask = torch.zeros_like(x).to(device=x.device, non_blocking=True)
    t_lengths = lengths[:,:,None].expand_as(mask)
    arange_id = torch.arange(mask.size(1)).to(device=x.device, non_blocking=True)
    arange_id = arange_id[None,:,None].expand_as(mask)

    mask[arange_id<t_lengths] = 1
    # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    # exp(x - max(x)) instead of exp(x) is a trick
    # to improve the numerical stability while giving
    # the same outputs
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class GradReverseMask(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    
    """
    @staticmethod
    def forward(ctx, x, mask, weight):
        """
        The mask should be composed of 0 or 1. 
        The '1' will get their gradient reversed..
        """
        ctx.save_for_backward(mask)
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        mask_c = mask.clone().detach().float()
        mask_c[mask == 0] = 1.0
        mask_c[mask == 1] = - float(ctx.weight)
        return grad_output * mask_c[:, None].float(), None, None


def grad_reverse_mask(x, mask, weight=1):
    return GradReverseMask.apply(x, mask, weight)


class GradReverse(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)



class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)