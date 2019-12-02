import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def getData(name='cifar10', train_bs=128, test_bs=1000):    
    
    
    if name == 'svhn':
        train_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='extra', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
    datasets.SVHN('../data', split='test', download=True,transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=test_bs, shuffle=False)
   
    
    
    
    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=test_bs, shuffle=False)


    if name == 'emnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('../data', train=True, download=True, split='balanced',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1751,), (0.3267,))
                           ])),
            batch_size=train_bs, shuffle=True)
    
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('../data', train=False, split='balanced', transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1751,), (0.3267,))
                           ])),
            batch_size=test_bs, shuffle=False)
    
    
    
    
    if name == 'cifar10':
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

        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    
    
    
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

        trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR100(root='../data', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)
    
    
    
    if name == 'tinyimagenet':      
        normalize = transforms.Normalize(mean=[0.44785526394844055, 0.41693055629730225, 0.36942949891090393],
                                     std=[0.2928885519504547, 0.28230994939804077, 0.2889912724494934])
        train_dataset = datasets.ImageFolder(
        '../data/tiny-imagenet-200/train',
        transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=False)
        
        test_dataset = datasets.ImageFolder(
        '../data/tiny-imagenet-200/val',
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
        
    return train_loader, test_loader





def exp_lr_scheduler(epoch, optimizer, strategy=True, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    print(strategy)

    if strategy=='normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer


