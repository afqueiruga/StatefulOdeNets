import torch
import torchvision



def get_dataset(name='FMNIST', batch_size=128, root='.'):
    if name=='CIFAR10':
        transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                         (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                         (0.2023, 0.1994, 0.2010)),
        ])        
        
        
        refset = torchvision.datasets.CIFAR10(root=root+'/CIFAR10_data', 
                    train=True, download=True, transform=None)
        trainset = torchvision.datasets.CIFAR10(root=root+'/CIFAR10_data', 
                    train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, 
                    batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR10(root=root+'/CIFAR10_data', 
                    train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, 
                    batch_size=batch_size, shuffle=True, num_workers=2)
        
    
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
        trainloader = torch.utils.data.DataLoader(trainset, 
                batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.FashionMNIST(root+'/F_MNIST_data/', 
                download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, 
                batch_size=batch_size, shuffle=True)
        
    else:
        raise RunetimeError('Unknown dataset')
        
    return refset,trainset,trainloader,testset,testloader