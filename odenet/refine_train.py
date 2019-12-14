import torch
from .odenet import refine
from .helper import get_device, which_device
import collections
import torch.nn.init as init
import copy


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
    


def train_adapt(model, loader, testloader, criterion, N_epochs, N_refine=[],
               lr=1.0e-3, lr_decay=0.2, epoch_update=[], weight_decay=1e-5, device=None):
    """I don't know how to control the learning rate"""

    if device is None:
        device = get_device()
    losses = []
    train_acc = []
    test_acc = []
    refine_steps = []
    model_list = [model]
    N_print= 1
    lr_init = lr
    lr_current = lr
    step_count = 0

    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(N_epochs):
        model.train()

        if e in N_refine:
            new_model = model.refine()
            model_list.append(new_model)
            model = new_model

            print('**** Setup ****')
            print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
            print('************')
            #print(model)
 
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_current, momentum=0.9, 
                                        weight_decay=weight_decay)

            optimizer.state = collections.defaultdict(dict) # Reset state

            
            refine_steps.append(step_count)        
        
        

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

        if e % 5 == 0:
            print('Epoch: ', e)
        
        if e % 5 == 0:
            model.eval()
            correct = 0
            total_num = 0        

            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_num += len(data)
                
            print('Train Accuracy: ', correct / total_num)    
            train_acc.append(correct / total_num)
            

        if e % 5 == 0:
            model.eval()
            correct = 0
            total_num = 0   
            
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_num += len(data)
            print('Test Accuracy: ', correct / total_num)        
            test_acc.append(correct / total_num)


        if e in epoch_update:
            lr_current *= lr_decay
            


        optimizer = exp_lr_scheduler(optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)

    return model_list, losses, refine_steps, train_acc, test_acc

#
# Evaluation tools
#
def acc(y,labels):
    return torch.sum(torch.argmax(y,dim=-1) == labels)*1.0/len(labels)
def model_acc(model,loader):
    try:
        imgs,labels = loader
    except:
        imgs,labels = next(iter(loader))
    y = model(imgs.to(which_device(model)))
    return acc(y.cpu(),labels)
def plot_accuracy(model,loader):
    try:
        imgs,labels = loader
    except:
        imgs,labels = next(iter(loader))
    y = model(imgs.to(which_device(model)))
    print(acc(y.cpu(),labels).item())
    bars = torch.nn.Softmax(dim=-1)(y[:10])
    size = len(bars)
    plt.figure(figsize=(10,10))
    for i,(pred,img,label) in enumerate(zip(bars,imgs,labels)):
        plt.subplot(size//2+1,4,1+2*i)
        plt.imshow(img[0,:,:].detach().numpy(),cmap='Greys')
        plt.subplot(size//2+1,4,2+2*i)
        plt.bar(range(10),[1 if y==label else 0 for y in range(10)])
        plt.bar(range(10),pred.cpu().detach().numpy())
    plt.show()

#
# Plotting tools
#
from matplotlib import pylab as plt
def plot_weights_over_time(model_list, grab_w, grab_ts):
    for i,m in enumerate(model_list):
        w = grab_w(m).detach().numpy()
        ts =  grab_ts(m).detach().numpy()
        #print(ts.shape, w.shape)
        #plt.imshow(w[:,0,:].T)
        plt.subplot(len(model_list),1,i+1)
        dt = ts[1]-ts[0]
        plt.bar(ts[0:-1]+dt*0.5,w,width=ts[1]-ts[0],edgecolor='k')
        plt.xlabel('t')
    plt.show()

def plot_layers_over_times(model, img):
    y = models.channel_squish(img,2)
    with torch.no_grad():
        yy = torchdiffeq.odeint(m.net[0].net , y, m.net[0].ts)
    plt.figure(figsize=(8,20))
    L = yy.shape[0]
    for i in range(L):
        for j in range(4):
            plt.subplot(L,4,4*i+j+1)
            plt.imshow(yy[i,2,j,:,:])
    plt.show()
   
    
#def train_for_epochs(model, loader, testloader,
#                     criterion,
#                     N_epochs, losses = None, 
#                     lr=1.0e-3, lr_decay=0.2, epoch_update=[], weight_decay=1e-5,
#                     N_print=1000, device=None):
#    "Works for normal models too"
#    if device is None:
#        device = get_device()
#    if losses is None:
#        losses = []
#    #criterion = torch.nn.BCEWithLogitsLoss()
#    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
#    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#    step_count = 0
#    for e in range(N_epochs):
#        for imgs,labels in iter(loader):
#            imgs = imgs.to(device)
#            labels = labels.to(device)
#            optimizer.zero_grad()
#            out = model(imgs)
#            L = criterion(out,labels)
#            L.backward()
#            optimizer.step()
#            losses.append(L.detach().cpu().item())
#            if step_count % N_print == N_print-1:
#                print(L.detach().cpu())
#            step_count += 1
#        
#        #exp_lr_scheduler(optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)
#        
#        model.eval()
#        correct = 0
#        total_num = 0
#        for data, target in loader:
#            data, target = data.to(device), target.to(device)
#            output = model(data)
#            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
#            total_num += len(data)
#        print('Train Loss: ', correct / total_num)         
#
#        for data, target in testloader:
#            data, target = data.to(device), target.to(device)
#            output = model(data)
#            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
#            total_num += len(data)
#        print('Test Loss: ', correct / total_num)         
#        
#        
#    return losses    
    