import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True

def num_parameters(module: torch.nn.Module):
    return sum([
        np.prod(x.shape) for x in module.parameters()
    ])


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


def format_params(module: torch.nn.Module):
    n = num_parameters(module)
    if n > 10**6:
        n /= 10**6
        return f"{n:.2f}M"
    if n > 10**3:
        n /= 10**3
        return f"{n:.1f}K"
    return n