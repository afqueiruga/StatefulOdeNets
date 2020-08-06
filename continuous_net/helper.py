import numpy as np
import torch
import torch.nn.init as init

#
# Torch helpers to keep environments uniform.
#
def set_seed(seed=1, deterministic=False):
    """Set one seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device_id=None):
    """Get a gpu if available."""
    if device_id is None:
        device_id = 'cuda:0'
    try:
        device = torch.device(device_id)
        print("Connected to device ", device_id)
    except:
        device = torch.device('cpu')
        print("Falling back to cpu")
    # if torch.cuda.device_count()>0:
    #     device = torch.device(f'cuda:{which_gpu}')
    #    print("Connected to a GPU")
    #else:
    #    device = torch.device('cpu')
    #    print("Using the CPU")
    return device

def which_device(model):
    return next(model.parameters()).device

