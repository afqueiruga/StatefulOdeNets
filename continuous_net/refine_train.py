import collections
from typing import List, Any
import timeit
import attr
import torch
import torch.nn.init as init

from .helper import get_device, which_device
from .ode_models import refine


@attr.s(auto_attribs=True)
class Result:
    """A container class to collect training state"""
    model_list: List[Any]
    losses: Any
    refine_steps: Any
    train_acc: Any
    test_acc: Any
    epoch_times: List[Any]


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
                refine_variance=0.0,
                device=None,
                fname=None,
                SAVE_DIR=None):
    """Adaptive Refinement Training for RefineNets"""
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
    epoch_times = []
    model_list = [model]
    lr_current = lr
    step_count = 0
    want_train_acc = False

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # Uncomment to use 4 gpus
    USE_PARALLEL = False
    if USE_PARALLEL:
        model_single = model
        model = torch.nn.DataParallel(model_single, device_ids=[0,1,2,3])

    for e in range(N_epochs):
        model.train()

        # Make a new model if the epoch number is in the schedule
        if e in N_refine:
            # Get back from parallel
            if USE_PARALLEL:
                model = model.module
            new_model = model.refine(refine_variance)
            model_list.append(new_model)
            # Make the new one parallel
            if USE_PARALLEL:
                model = torch.nn.DataParallel(new_model, device_ids=[0,1,2,3])
            else:
                model = new_model
            print('**** Allocated refinment ****')
            print('Total params: %.2fk' % (count_parameters(model)/1000.0))
            print('************')
            te_acc = calculate_accuracy(model, testloader)
            print('Test Accuracy after refinement: ', te_acc)
            test_acc.append( (e,te_acc) )
            if want_train_acc:
                tr_acc = calculate_accuracy(model, loader)
                print('Train Accuracy after refinement: ', tr_acc)
                train_acc.append( (e,tr_acc) )
            print(model)
            # We need to reset the optimizer to point to the new weights
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_current, momentum=0.9, weight_decay=weight_decay)
            refine_steps.append(step_count)
            
        starting_time = timeit.default_timer()
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
            if torch.isnan(L):
                print("Hit a NaN, returning early.")
                return Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
            _loss = L.detach().cpu().item()
            losses.append(_loss)
            step_count+=1
        epoch_times.append(timeit.default_timer() - starting_time)
        #print("Epoch took ", epoch_times[-1], " seconds.")




        # Evaluate training and testing accuracy
        n_print = 5
        if e == 0 or (e+1) % n_print == 0:
            print('After Epoch: ', e)
            model.eval()
            if want_train_acc:
                tr_acc = calculate_accuracy(model, loader)
                print('Train Accuracy: ', tr_acc)
                train_acc.append( (e,tr_acc) )
            te_acc = calculate_accuracy(model, testloader)
            print('Test Accuracy: ', te_acc)
            test_acc.append( (e,te_acc) )
        # Save checkpoint
        if fname is not None and SAVE_DIR is not None and (e+1)%n_print==0:
            chckpt = Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
            try:
                os.mkdir(SAVE_DIR)
                print("Making directory ", SAVE_DIR)
            except:
                print("Directory ", SAVE_DIR, " already exists.")
            torch.save(chckpt, fname+f"-CHECKPOINT-{e}.pkl")

        # learnin rate schedule
        if e in epoch_update:
            lr_current *= lr_decay

        optimizer = exp_lr_scheduler(
            optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)
        
        
    print('Average time (without eval): ', np.mean(epoch_times))
    print('Total time (without eval): ', np.sum(epoch_times))        
    return Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
