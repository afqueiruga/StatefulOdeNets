"""
RefineNet
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq

from .helper import which_device


def refine(net):
    try:
        return net.refine()
    except AttributeError:
        if type(net) is torch.nn.Sequential:
            return torch.nn.Sequential(*[
                refine(m) for m in net
            ])
        else:
            #raise RuntimeError("Hit a network that cannot be refined.")
            # Error is for debugging. This makes sense too:
            return copy.deepcopy(net)


class LinearODE(torch.nn.Module):
    def __init__(self, time_d, in_features, out_features):
        super(LinearODE,self).__init__()
        self.time_d = time_d
        self.out_features = out_features
        self.in_features = in_features
        self.weights = torch.nn.Parameter(torch.randn(time_d, in_features, out_features) / (out_features)**0.5)
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_features))
        
    def forward(self, t, x):
        # Use the trick where it's the same as index selection
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        wij = self.weights[t_idx,:,:]
        bi = self.bias[t_idx,:]
        y = x @ wij+ bi # TODO use torch.linear
        return y
    
    def refine(self):
        new = LinearODE(2*self.time_d,
                       self.in_features,
                       self.out_features)
        for t in range(self.time_d):
            new.weights.data[2*t:2*t+2,:,:] = self.weights.data[t,:,:]
        for t in range(self.time_d):
            new.bias.data[2*t:2*t+2,:] = self.bias.data[t,:]
        return new

    
class ShallowODE(torch.nn.Module):
    def __init__(self, time_d, in_features, hidden_features, 
                 act=torch.nn.functional.relu):
        super(ShallowODE,self).__init__()
        self.act = act
        self.L1 = LinearODE(time_d,in_features,hidden_features)
        self.L2 = LinearODE(time_d,hidden_features,in_features)
    def forward(self, t, x):
        h = self.L1(t,x)
        hh = self.act(h)
        y = self.L2(t,hh)
        yy = self.act(y)
        return yy
    def refine(self):
        L1 = self.L1.refine()
        L2 = self.L2.refine()
        # TODO Don't like it, it re-allocates the weights that we're gonna throw away
        new = copy.deepcopy(self)
        new.L1 = L1
        new.L2 = L2
        return new
    
    
class Conv2DODE(torch.nn.Module):
    def __init__(self, time_d, in_channels, out_channels,
                width=1, padding=1):
        super(Conv2DODE,self).__init__()
        self.time_d = time_d
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.width = width
        self.padding = padding
        self.weights = torch.nn.Parameter(torch.randn(time_d, out_channels,in_channels, width , width) / (out_channels)**0.5)
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_channels))
        
    def forward(self, t, x):
        # Use the trick where it's the same as index selection
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        #t_idx = torch.LongTensor([t_idx]).to(which_device(self))
        wij = self.weights[t_idx,:,:,:,:]
        bi = self.bias[t_idx,:]
        y = torch.nn.functional.conv2d(x, wij,bi, padding=self.padding)
        return y
    
    def refine(self):
        new = Conv2DODE(2*self.time_d,
                       self.in_channels,
                       self.out_channels,
                       width=self.width,
                       padding=self.padding).to(which_device(self))
        for t in range(self.time_d):
            new.weights.data[2*t:2*t+2,:,:,:,:] = self.weights.data[t,:,:,:,:]
        for t in range(self.time_d):
            new.bias.data[2*t:2*t+2,:] = self.bias.data[t,:]
        return new

# TODO batchnorms are still messed up
    
class ShallowConv2DODE(torch.nn.Module):
    def __init__(self, time_d, in_features, hidden_features, 
                 width=3, padding=1,
                 act=torch.nn.functional.relu,
                 use_batch_norms=True):
        super().__init__()
        self.act = act
        self.use_batch_norms=use_batch_norms
        self.verbose=False
        self.L1 = Conv2DODE(time_d,in_features,hidden_features,
                            width=width, padding=padding)
        
        self.L2 = Conv2DODE(time_d,hidden_features,in_features,
                            width=width, padding=padding)
        
        if use_batch_norms:
            self.bn1 = torch.nn.BatchNorm2d(
                hidden_features, affine=True, track_running_stats=True)
            self.bn2 = torch.nn.BatchNorm2d(
                in_features, affine=True, track_running_stats=True)
        
    def forward(self, t, x):
        if self.verbose: print("shallow @ ",t)
  
        x = self.L1(t, x)
        x = self.act(x)
        if self.use_batch_norms:
            x = self.bn1(x)
        x = self.L2(t, x)
        x = self.act(x)
        if self.use_batch_norms:
            x = self.bn2(x)
        return x
    
    def refine(self):
        #with torch.no_grad():
            L1 = self.L1.refine()
            L2 = self.L2.refine()
            
            if self.use_batch_norms:
                self.bn1.track_running_stats = False
                self.bn2.track_running_stats = False

            # TODO Don't like it, it re-allocates the weights that we're gonna throw away
            new = copy.deepcopy(self) 
            new.L1 = L1
            new.L2 = L2
            
            #new.bn1.reset_running_stats()
            #new.bn2.reset_running_stats()
 
#            new.bn1.track_running_stats = False
#            new.bn2.track_running_stats = False                
#            new.bn1.affine = False
#            new.bn2.affine = False
            if self.use_batch_norms:
                self.bn1.track_running_stats = True
                self.bn2.track_running_stats = True
            return new

#        new.bn1.reset_running_stats()
#        new.bn1.register_buffer('running_mean', torch.zeros(new.bn1.num_features))
#        new.bn1.register_buffer('running_var', torch.ones(new.bn1.num_features))
#        new.bn1.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#
#        new.bn2.reset_running_stats()
#        new.bn2.register_buffer('running_mean', torch.zeros(new.bn2.num_features))
#        new.bn2.register_buffer('running_var', torch.ones(new.bn2.num_features))
#        new.bn2.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#  
        
#        new.bn1 = torch.nn.BatchNorm2d(self.bn1.num_features, track_running_stats=True)
#        new.bn1.weight.data = self.bn1.weight.data
#        new.bn1.bias.data = self.bn1.bias.data
#        new.bn1.running_mean.data = self.bn1.running_mean.data
#        new.bn1.running_var.data = self.bn1.running_var.data
#        new.bn1.num_batches_tracked.data = self.bn1.num_batches_tracked.data
#
#        
#        
#        new.bn2 = torch.nn.BatchNorm2d(self.bn1.num_features, track_running_stats=True)
#        new.bn2.weight.data = self.bn2.weight.data
#        new.bn2.bias.data = self.bn2.bias.data
#        new.bn2.running_mean.data = self.bn2.running_mean.data
#        new.bn2.running_var.data = self.bn2.running_var.data
#        new.bn2.num_batches_tracked.data = self.bn2.num_batches_tracked.data

        


class ODEify(torch.nn.Module):
    """Throws away the t."""
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self,t,x):
        return f(x)
    def refine(self):
        return ODEify(f)
    

class ODEBlock(torch.nn.Module):
    """Wraps an ode-model with the odesolve to fit into standard 
    models."""
    def __init__(self,net,N_time=1,method='euler',use_adjoint=False):
        super(ODEBlock,self).__init__()
        self.N_time = N_time
        self.method = method
        self.use_adjoint = use_adjoint
        self.ts = torch.linspace(0,1,N_time+1) # TODO: awk with piecewise constant centered on the half-cells
        self.net = net
    
    def forward(self,x):
        if self.use_adjoint:
            integ = torchdiffeq.odeint_adjoint
        else:
            integ = torchdiffeq.odeint
        h = integ(self.net, x, self.ts, method=self.method,
                  options=dict(enforce_openset=True))[-1,:,:]
        return h
    
    def refine(self):
        newnet = self.net.refine()
        new = ODEBlock(newnet,self.N_time*2,method=self.method).to(which_device(self))
        return new
    
    def diffeq(self,x):
        hs = torchdiffeq.odeint(self.net, x, self.ts, method=self.method,
                              options=dict(enforce_openset=True))
        return hs
    
