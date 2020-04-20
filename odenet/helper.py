import numpy as np
import torch
import torch.nn.init as init

#
# Torch helpers to keep environments uniform.
#
def set_seed(seed=1):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(which_gpu=0):
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device(f'cuda:{which_gpu}')
        print("Connected to a GPU")
    else:
        device = torch.device('cpu')
        print("Using the CPU")
    return device

def which_device(model):
    return next(model.parameters()).device

