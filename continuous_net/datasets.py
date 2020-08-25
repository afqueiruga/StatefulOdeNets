import torch
import torchvision
from torchvision import datasets, transforms

def get_dataset(name='FMNIST', batch_size=128, test_batch_size=256, root='.', device=None):

    if name=='CIFAR10':
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

        refset = datasets.CIFAR10(root=root+'/CIFAR10_data',
                    train=True, download=True, transform=None)
        trainset = datasets.CIFAR10(root=root+'/CIFAR10_data',
                    train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=root+'/CIFAR10_data',
                    train=False, download=True, transform=transform_test)

    elif name=='FMNIST':
        transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,),
                                                 (0.5,)),
            ])
        refset = torchvision.datasets.FashionMNIST(root+'/F_MNIST_data/',
                download=True, train=True, transform=None)
        trainset = torchvision.datasets.FashionMNIST(root+'/F_MNIST_data/',
                download=True, train=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root+'/F_MNIST_data/',
                download=True, train=False, transform=transform)

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

        trainset = datasets.CIFAR100(
            root=root+'/CIFAR100/', train=True, download=True,
            transform=transform_train)

        testset = datasets.CIFAR100(
            root=root+'/CIFAR100/', train=False, download=False,
            transform=transform_test)
        refset = None

    elif name == 'tinyimagenet':
        normalize = transforms.Normalize(
            mean=[0.44785526394844055, 0.41693055629730225, 0.36942949891090393],
            std=[0.2928885519504547, 0.28230994939804077, 0.2889912724494934])
        trainset = datasets.ImageFolder(
            root+'/tiny-imagenet-200/train',
            transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=False)
        testset = datasets.ImageFolder(
            root+'/tiny-imagenet-200/val',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs, shuffle=False)
        refset = None

    else:
        raise RuntimeError('Unknown dataset')



    if device is not None:
        trainset = trainset.to(device)
        testnset = testset.to(device)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=True, num_workers=2,
        pin_memory=True)

    return refset,trainset,trainloader,testset,testloader
