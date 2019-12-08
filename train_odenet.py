import torch
from matplotlib import pylab as plt
from odenet import datasets
from odenet.odenet_cifar10 import ODEResNet
from odenet import refine_train

import importlib
importlib.reload(refine_train)
#importlib.reload(odenet_cifar10)


def run_train(method,initial_depth,ALPHA=16,
              epochs_per_level=1,adapt=1,
              lr=1.0e-3,lr_decay=0.2):
    refset,trainset,trainloader,testset,testloader \
        = datasets.get_dataset('CIFAR10')


    model = ODEResNet(ALPHA=ALPHA,method=method,time_d=time_d, in_channels=3)
    RES = refine_train.train_adapt(model, trainloader,
                                   torch.nn.CrossEntropyLoss(),
                                   epochs_per_level, adapt, 
                                   lr=lr, lr_decay=lr_decay)

    model_list,losses,refine_steps = RES
    return model_list,losses,refine_steps

methods = ['euler','midpoint','rk4']

#run_train(method,)