import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder


def get_dataset(name='CIFAR10',
                batch_size=128,
                test_batch_size=256,
                root='.',
                device=None,
                seed=0):

    if name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        refset = CIFAR10(root=root + '/CIFAR10_data',
                         train=True,
                         download=True,
                         transform=transform_test)
        trainset = CIFAR10(root=root + '/CIFAR10_data',
                           train=True,
                           download=True,
                           transform=transform_train)
        testset = CIFAR10(root=root + '/CIFAR10_data',
                          train=False,
                          download=True,
                          transform=transform_test)
        
    elif name == 'FMNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        refset = FashionMNIST(root + '/F_MNIST_data/',
                              download=True,
                              train=True,
                              transform=transform)
        trainset = FashionMNIST(root + '/F_MNIST_data/',
                                download=True,
                                train=True,
                                transform=transform)
        testset = FashionMNIST(root + '/F_MNIST_data/',
                               download=True,
                               train=False,
                               transform=transform)

    elif name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = CIFAR100(root=root + '/CIFAR100/',
                            train=True,
                            download=True,
                            transform=transform_train)
        testset = CIFAR100(root=root + '/CIFAR100/',
                           train=False,
                           download=False,
                           transform=transform_test)
        refset = None


    else:
        raise RuntimeError('Unknown dataset')

    n_dataset = len(trainset)
    n_train = int(1 * n_dataset)
    n_val = n_dataset - n_train
    trainset, validationset = torch.utils.data.random_split(
            trainset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed))
    
    if device is not None:
        trainset = trainset.to(device)
        validationset = refset.to(device)
        testnset = testset.to(device)

    train_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=test_batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

    return train_loader, validation_loader, test_loader
    
