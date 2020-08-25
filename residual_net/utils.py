import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


#from continuous_net.odenet_cifar10 import *
from continuous_net.helper import set_seed, get_device, which_device

device = get_device()


def getData(name='cifar10', batch_size=128, test_batch_size=512):


    if name == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        transform_test = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)



    if name == 'cifar100':

        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

        testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)



    if name == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[0.44785526394844055, 0.41693055629730225, 0.36942949891090393],
                                     std=[0.2928885519504547, 0.28230994939804077, 0.2889912724494934])
        train_dataset = datasets.ImageFolder(
        './data/tiny-imagenet-200/train',
        transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

        test_dataset = datasets.ImageFolder(
        './data/tiny-imagenet-200/val',
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader





def exp_lr_scheduler(epoch, optimizer, strategy=True, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""

    if strategy=='normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer
