import numpy as np
import torch
import torch.nn.init as init

#
# Torch helpers to keep environments uniform.
#
def set_seed():
    """Set one seed for reproducibility."""
    np.random.seed(10)
    torch.manual_seed(10)

def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device


def which_device(model):
    return next(model.parameters()).device