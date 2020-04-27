"""This is the top level training program. It can be scripted in python or run from the CLI through cli.py."""
import os
from matplotlib import pylab as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from odenet import datasets
from odenet.helper import set_seed, get_device, which_device
from odenet.odenet_cifar10 import ODEResNet
from odenet import refine_train


def file_name(method):
    return 'results/resnet_' + method + '.pkl'

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

def do_a_train_set(
    dataset, ALPHA, scheme, N_epochs, N_adapt, lr,
    lr_decay=0.1, epoch_update=[10], weight_decay=1e-5,               
    seed=None, device=None):
    """Set up and train one model, and save it.
    
    Args:
        dataset: Which dataset to load
        ALPHA: Initial refinement
        scheme: One of "euler", "midpoint", or "RK4"
        N_epochs: How many dataset epochs to train over for the whole duration
        N_adapt: A list of epoch numbers at which to refine. Only numbers less than N_epochs are meaningful
        lr: Initial learning rate
        lr_decay: Learning rate decay
        epoch_update: A list of epochs at which to call a learning_rate schedule
        weight_decay: Traditional weight decay parameter
        seed: a seed
        device: which device to use
    """
    try:
        os.mkdir('results')
        print("Making directory ", "results.")
    except:
        print("Directory ", "results", " already exists.")
    
    set_seed(seed)
    device = get_device(device)

    refset,trainset,trainloader,testset,testloader = datasets.get_dataset(dataset,root='../data/')

    if dataset=="CIFAR10":
        in_channels=3
    elif dataset=="FMNIST":
        in_channels=1
    else:
        in_channels=3
    model = ODEResNet(ALPHA=ALPHA, method=scheme, in_channels=in_channels).to(device)
    #model.apply(init_params)
    #model = torch.nn.DataParallel(model)
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print(model)
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    
    res = refine_train.train_adapt(
        model, trainloader, testloader, torch.nn.CrossEntropyLoss(),
        N_epochs, N_adapt, lr=lr, lr_decay=lr_decay, epoch_update=epoch_update, weight_decay=weight_decay,
        device=device)
    torch.save(res, file_name(scheme))

    #plt.semilogy(res[1])
    #for r in res[2]:
    #    plt.axvline(r,color='k')
    return res
