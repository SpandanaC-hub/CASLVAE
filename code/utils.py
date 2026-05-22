import datetime
import torch
import torch.nn.functional as F
import numpy as np

def sample_gumbel_softmax(logits, tau=1.0, hard=False):
    """ Samples from a categorical distribution using the Gumbel-Softmax reparameterization trick. 
    Args: 
        logits (torch.Tensor): Unnormalized log-probabilities for each category. 
        tau (float, optional): Temperature parameter controlling the smoothness of the distribution. 
                                Lower values make the output closer to a one-hot vector. Default is 1.0. 
        hard (bool, optional): If True, returns a one-hot vector using the straight-through estimator 
                                (hard in the forward pass, soft in the backward pass). If False, returns a differentiable soft sample. 
                                Default is False. 
        Returns: torch.Tensor: A differentiable sample lying on the probability simplex. 
                                If `hard=True`, the returned tensor is one-hot in the forward pass but maintains gradients as if it were soft. 
    """
    gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
    y = F.softmax((logits + gumbel) / tau, dim=-1)

    if hard:
        # Straight-through: take one-hot in forward, soft in backward
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = (y_hard - y).detach() + y
    return y

def mmd(x, y, gammas, device='cpu'):
    """Computes the Maximum Mean Discrepancy (MMD) between two samples x and y using a multi-kernel RBF estimator.
    MMD measures how different two distributions are by comparing expectations
    of kernel functions. A value of 0 indicates identical distributions.
    Args:
        x,y (torch.Tensor):
            x is first sample set of shape (N, D). y is second sample set of shape (M, D).
        gammas (torch.Tensor): 1D tensor of RBF kernel bandwidths. Multiple gammas implement
            a multi-kernel MMD for better stability.
        device (torch.device):
            Device on which computations should run.
    Returns: torch.Tensor: Scalar MMD value. Guaranteed to be non-negative (numerical
            underflow is clipped to zero).
    """
    # Ensure gamma bandwidths are on the correct device
    gammas = gammas.to(device)
    # Compute kernel expectations:  E[k(x, x')] + E[k(y, y')] - 2E[k(x, y')]           
    cost = torch.mean(gram_matrix(x, x, gammas=gammas)).to(device)
    cost += torch.mean(gram_matrix(y, y, gammas=gammas)).to(device)
    cost -= 2 * torch.mean(gram_matrix(x, y, gammas=gammas)).to(device)
    # Numerical safety: MMD should never be negative
    if cost < 0:
        return torch.tensor(0).to(device)
    return cost

def gram_matrix(x, y, gammas):
    """
    Computes a multi-kernel RBF Gram matrix between samples x and y.
    Each gamma defines an RBF kernel:
        k(x, y) = exp(-gamma * ||x - y||^2)
    The final kernel value is the sum over all gammas.
    """
    gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0) # Compute pairwise Euclidean distances between all x and y samples
    pairwise_distances_sq = torch.square(pairwise_distances)     # Square distances for the RBF kernel

    # Apply all gamma bandwidths: # tmp = exp(-gamma * ||x - y||^2) summed over all gammas 
    # Flatten distances → multiply by gammas → reshape back
    tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape)
    return tmp

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)
        if m.bias is not None:
            m.bias.data.normal_(0, 0.1)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)