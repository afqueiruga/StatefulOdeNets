##############################################################################
## Attacks written by Zhewei Yao <zheweiy@berkeley.edu>
#############################################################################

import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad

import numpy as np

from copy import deepcopy
#==============================================================================
## FGSM
#==============================================================================
def fgsm(model, data, target, eps):
    """Generate an adversarial pertubation using the fast gradient sign method.

    Args:
        data: input image to perturb
    """
    #model.eval()
    data, target = Variable(data.cuda(), requires_grad=True), target.cuda()
    #data.requires_grad = True
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward(create_graph=False)
    pertubation = eps * torch.sign(data.grad.data)
    x_fgsm = data.data + pertubation
    X_adv = torch.clamp(x_fgsm, torch.min(data.data), torch.max(data.data))

    return X_adv


def fgsm_iter(model, data, target, eps, iterations=10):
    """
    iteration version of fgsm
    """
    
    X_adv = fgsm(model, data, target, eps)
    for i in range(iterations):
    	X_adv = fgsm(model, X_adv, target, eps)
        
    return X_adv




def fgsm_adaptive_iter(model, data, target, eps, iterations):
    update_num = 0
    i = 0
    while True:
        if i >= iterations:
            #print('failed to fool all the image')
            data = Variable(data)
            break
        
        #model.eval()
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        tmp_mask = pred.view_as(target) == target.data # get index
        update_num += torch.sum(tmp_mask.long())
        
        #print(torch.sum(tmp_mask.long()))
        if torch.sum(tmp_mask.long()) < 1: # allowed fail
            break
        
        attack_mask = tmp_mask.nonzero().view(-1)
        data[attack_mask,:] = fgsm(model, data[attack_mask,:], target[attack_mask], eps)
        #data = fgsm(model, data, target, eps)
        
        i += 1
        
    return data.data, update_num








#==============================================================================
## Deep Fool
#==============================================================================

def deep_fool(model, data, c=9, p=2):
    """Generate an adversarial pertubation using the dp method.

    Args:
        data: input image to perturb
    """
    #model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    
    output, ind = torch.sort(output, descending=True)
    #c = output.size()[1]
    n = len(data)

    true_out = output[range(len(data)), n*[0]]
    z_true = torch.sum(true_out)
    data.grad = None
    z_true.backward(retain_graph=True)
    true_grad = data.grad
    grads = torch.zeros([1+c] + list(data.size())).cuda()
    pers = torch.zeros(len(data), 1+c).cuda()
    for i in range(1,1+c):
        z = torch.sum(output[:,i])
        data.grad = None
        model.zero_grad()
        z.backward(retain_graph=True)
        grad = data.grad # batch_size x 3k
        grads[i] = grad.data
        grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
        pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
    pers[range(n),n*[0]] = np.inf
    pers[pers < 0] = 0
    per, index = torch.min(pers,1) # batch_size x 1
    #print('maximum pert: ', torch.max(per))
    update = grads[index,range(len(data)),:] - true_grad.data
    
    if p == 1:
        update = torch.sign(update)
    
    elif p ==2:
        update = update.view(n,-1)
        update = update / (torch.norm(update, p=2, dim=1).view(n,1)+1e-6)
    X_adv = data.data + torch.diag(torch.abs((per+1e-4)*1.02)).mm(update.view(n,-1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv



def deep_fool_iter(model, data, target, c=9, p=2, iterations=10):
    X_adv = data.cuda() + 0.0
    update_num = 0.
    for i in range(iterations):
        #model.eval()
        Xdata, Xtarget = X_adv, target.cuda()
        Xdata, Xtarget = Variable(Xdata, requires_grad=True), Variable(Xtarget)
        model.zero_grad()
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Xtarget)==Xtarget.data # get index
        update_num += torch.sum(tmp_mask.long())
        #print('need to attack: ', torch.sum(tmp_mask))
        if torch.sum(tmp_mask.long()) < 1:
            break
        #print (i, ': ', torch.sum(tmp_mask.long()))
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask,:] = deep_fool(model, X_adv[attack_mask,:], c=c, p=p)
    model.zero_grad()
    return X_adv, update_num



#################################################
## Select Attack Index 
#################################################
def select_index(model, data, c=9, p=2, worst_case = False):
    """Generate an adversarial pertubation using the dp method.

    Args:
        data: input image to perturb
    """
    #model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    #output = F.softmax(output) 
    output, ind = torch.sort(output, descending=True)
    n = len(data)

    true_out = output[range(len(data)), n*[0]]
    z_true = torch.sum(true_out)
    data.grad = None
    z_true.backward(retain_graph=True)
    true_grad = data.grad
    pers = torch.zeros(len(data), 1+c).cuda()
    for i in range(1,1+c):
        z = torch.sum(output[:,i])
        data.grad = None
        model.zero_grad()
        z.backward(retain_graph=True)
        grad = data.grad # batch_size x 3k
        grad_diff = torch.norm(grad.data.view(n,-1) - true_grad.data.view(n,-1),p=p,dim=1) # batch_size x 1
        pers[:,i] = (true_out.data - output[:,i].data)/grad_diff # batch_size x 1
    if not worst_case:
        pers[range(n),n*[0]] = np.inf
        pers[pers < 0] = 0
        per, index = torch.min(pers,1) # batch_size x 1
    else:
        pers[range(n),n*[0]] = -np.inf
        per, index = torch.max(pers,1) # batch_size x 1
    
    output = []
    for i in range(data.size(0)):
        output.append(ind[i, index[i]].item())
    return torch.LongTensor(output) 



#################################################
## TR First Order Attack
#################################################
def tr_attack(model, data, true_ind, target_ind, eps, p = 2):
    """Generate an adversarial pertubation using the TR method.
    Pick the top false label and perturb towards that.
    First-order attack

    Args:
        data: input image to perturb
        true_ind: is true label
        target_ind: is the attack label
    """
    #model.eval()
    data = data.cuda()
    data.requires_grad = True
    model.zero_grad()
    output = model(data)
    n = len(data)

    q = 2
    if p == 8:
        q = 1
    
    output_g = output[range(n), target_ind] - output[range(n), true_ind]
    z = torch.sum(output_g)

    data.grad = None
    model.zero_grad()
    z.backward()
    update = deepcopy(data.grad.data) 
    update = update.view(n,-1)
    per = (-output_g.data.view(n,-1) + 0.) / (torch.norm(update, p = q, dim = 1).view(n, 1) + 1e-6)

    if p == 8 or p == 1:
        update = torch.sign(update)
    elif p ==2:
        update = update.view(n, -1)
        update = update / (torch.norm(update, p = 2, dim = 1).view(n,1) + 1e-6)
    per = per.view(-1)
    per_mask = per > eps
    per_mask = per_mask.nonzero().view(-1)
    # set overshoot for small pers
    per[per_mask] = eps
    X_adv = data.data + (((per + 1e-4) * 1.02).view(n,-1) * update.view(n, -1)).view(data.size())
    X_adv = torch.clamp(X_adv, torch.min(data.data), torch.max(data.data))
    return X_adv

            
def tr_attack_iter(model, data, target, eps, c = 9, p = 2, iterations = 1000, worst_case = False):
    X_adv = deepcopy(data.cuda()) 
    target_ind = select_index(model, data, c = c,p = p, worst_case = worst_case) 
    
    update_num = 0.
    for i in range(iterations):
        #model.eval()
        Xdata, Ytarget = X_adv, target.cuda()
        # First check if the input is correctly classfied before attack
        Xoutput = model(Xdata)
        Xpred = Xoutput.data.max(1, keepdim = True)[1] # get the index of the max log-probability
        tmp_mask = Xpred.view_as(Ytarget) == Ytarget.data # get index
        update_num += torch.sum(tmp_mask.long())
         # if all images are incorrectly classfied the attack is successful and exit
        if torch.sum(tmp_mask.long()) < 1:
            return X_adv, update_num      
        attack_mask = tmp_mask.nonzero().view(-1)
        X_adv[attack_mask,:]  = tr_attack(model, X_adv[attack_mask,:], target[attack_mask], target_ind[attack_mask], eps, p = p)
    return X_adv, update_num      
