import torch
import random
import numpy as np

def set_device():
    """
    Set the device to cuda if available, to mps if available, otherwise to cpu.
    
    Returns
    -------
    device: torch.device
        Device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def set_seed(seed):
    """
    Use this to set ALL the random seeds to
    a fixed value and take out any randomness from cuda kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False

    return True
