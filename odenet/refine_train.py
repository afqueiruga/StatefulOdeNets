import collections
from typing import List, Any
import attr
import torch
import torch.nn.init as init
import pytorch_memlab

from .helper import get_device, which_device
from .ode_models import refine

@attr.s(auto_attribs=True)
class Result:
    """A container class to collect tuples"""
    model_list: List[Any]
    losses: Any
    refine_steps: Any
    train_acc: Any
    test_acc: Any

#
# helper functions to examine models
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate
            print('lr decay update', param_group['lr'])
        return optimizer
    else:
        return optimizer  

def reset_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('reset lr', param_group['lr'])
    return optimizer

@torch.no_grad()
def calculate_accuracy(model, loader):
    device = which_device(model)
    correct, total_num = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    return correct / total_num


def train_adapt(model,
                loader,
                testloader,
                criterion,
                N_epochs,
                N_refine=None,
                lr=1.0e-3,
                lr_decay=0.2,
                epoch_update=None,
                weight_decay=1e-5,
                device=None):
    """I don't know how to control the learning rate"""
    if N_refine is None:
        N_refine = []
    if epoch_update is None:
        epoch_update = []
    if device is None:
        device = which_device(model)
    losses = []
    train_acc = []
    test_acc = []
    refine_steps = []
    model_list = [model]
    N_print= 1
    lr_init = lr
    lr_current = lr
    step_count = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    print('sgd')
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Random initialization checking accuracy metrics:')
    model.eval()
    tr_acc = calculate_accuracy(model, loader)
    print('Train Accuracy: ', tr_acc)
    train_acc.append(tr_acc)
    te_acc = calculate_accuracy(model, testloader)
    print('Test Accuracy: ', te_acc)
    test_acc.append(te_acc)
    memory_profile = pytorch_memlab.MemReporter(model)
    for e in range(N_epochs):
        model.train()
        
        # Make a new model if the epoch number is in the schedule
        if e in N_refine:
            memory_profile.report()
            new_model = model.refine()
            model_list.append(new_model)
            model = new_model
            print('**** Allocated refinment ****')
            print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
            print('************')
            print(model)
            
#            torch.save(model.state_dict(), 'temp' + '.pkl')                    
#            model = ODEResNet2(method='euler')    
#
#            device = get_device()
#            model.load_state_dict(torch.load('temp.pkl'))
#            model.to(device = device)
#            model.train() 
             # We need to reset the optimizer to point to the new weights
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_current, momentum=0.9, weight_decay=weight_decay)
            # Reset state
            # optimizer.state = collections.defaultdict(dict) 
            refine_steps.append(step_count)        
        
        # Train one epoch over the new model
        model.train()
        for imgs,labels in iter(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            L = criterion(out,labels)
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(L.detach().cpu().item())
            step_count+=1

        # Evaluate training and testing accuracy
        n_print = 1
        if e == 0 or (e+1) % n_print == 0:
            print('After Epoch: ', e)
            model.eval()
            # tr_acc = calculate_accuracy(model, loader)
            # print('Train Accuracy: ', tr_acc)
            # train_acc.append(tr_acc)
            te_acc = calculate_accuracy(model, testloader)
            print('Test Accuracy: ', te_acc)
            test_acc.append(te_acc)

        if e in epoch_update:
            lr_current *= lr_decay

        optimizer = exp_lr_scheduler(
            optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)
    
    memory_profile.report()
    return Result(model_list, losses, refine_steps, train_acc, test_acc)


def train_for_epochs(model,
                     loader,
                     testloader,
                     criterion,
                     N_epochs,
                     losses=None,
                     lr=1.0e-3,
                     lr_decay=0.2,
                     epoch_update=None,
                     weight_decay=1e-5,
                     N_print=1000,
                     device=None):
    """A training loop without refinement. Works for normal models too."""
    if epoch_update is None:
        epoch_update = []
    if device is None:
        device = which_device(model)
    if losses is None:
        losses = []
    train_acc = []
    test_acc = []
    #criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    step_count = 0
    for e in range(N_epochs):
        for imgs,labels in iter(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            L = criterion(out,labels)
            L.backward()
            optimizer.step()
            losses.append(L.detach().cpu().item())
            if step_count % N_print == N_print-1:
                print(L.detach().cpu())
            step_count += 1
        # exp_lr_scheduler(optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)   
        model.eval()
        correct = 0
        total_num = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
        print('Train Loss: ', correct / total_num)         
        train_acc.append( correct / total_num)
        
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
        print('Test Loss: ', correct / total_num)         
        test_acc.append( correct / total_num)

    return Result([model], losses, [], train_acc, test_acc)