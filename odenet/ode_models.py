"""
RefineNet
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq

from .helper import which_device


def piecewise_index(t, time_d):
    t_idx = int(t*time_d)
    if t_idx==time_d: t_idx = time_d-1
    return t_idx

def refine(net, variance=0.0):
    try:
        return net.refine(variance)
    except AttributeError as e:
        if type(net) is torch.nn.Sequential:
            return torch.nn.Sequential(*[
                refine(m, variance) for m in net
            ])
        if type(net) in (nn.Conv2d, nn.ReLU, nn.Flatten, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear, nn.BatchNorm2d):
            return copy.deepcopy(net)
        else:
            #raise RuntimeError("Hit a network that cannot be refined.")
            print("Could not refine ", net)
            # Error is for debugging. This makes sense too:
            raise e
            # return copy.deepcopy(net)


class LinearODE(torch.nn.Module):
    def __init__(self, time_d, in_features, out_features):
        super().__init__()
        self.time_d = time_d
        self.out_features = out_features
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.randn(time_d, in_features, out_features) / (out_features)**0.5)
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_features))
        
    def forward(self, t, x):
        # Use the trick where it's the same as index selection
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        wij = self.weight[t_idx,:,:]
        bi = self.bias[t_idx,:]
        y = x @ wij+ bi # TODO use torch.linear
        return y
    
    def refine(self, variance=0.0):
        new = LinearODE(2*self.time_d,
                       self.in_features,
                       self.out_features)
        for t in range(self.time_d):
            new.weight.data[2*t:2*t+2,:,:] = self.weight.data[t,:,:]
        for t in range(self.time_d):
            new.bias.data[2*t:2*t+2,:] = self.bias.data[t,:]
        if variance != 0:
            new.weight.data *= 1.0 + variance * torch.randn_like(new.weight)
            new.bias.data *= 1.0 + variance * torch.randn_like(new.bias)
        return new

class SkipInitODE(nn.Module):
    def __init__(self, time_d):
        super().__init__()
        self.time_d = time_d
        self.weight = nn.Parameter(torch.zeros(time_d).float())
    def forward(self, t, x):
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        #print(t_idx, t.data, self.weight[t_idx].data)
        return self.weight[t_idx] * x
    def refine(self, variance=0.0):
        new = SkipInitODE(2*self.time_d)
        for t in range(self.time_d):
            new.weight.data[2*t:2*t+2] = self.weight.data[t]
        return new
    
    
class ShallowODE(torch.nn.Module):
    def __init__(self, time_d, in_features, hidden_features, 
                 act=torch.nn.functional.relu):
        super().__init__()
        self.act = act
        self.L1 = LinearODE(time_d, in_features, hidden_features)
        self.L2 = LinearODE(time_d, hidden_features, in_features)
    def forward(self, t, x):
        h = self.L1(t, x)
        hh = self.act(h)
        y = self.L2(t, hh)
        yy = self.act(y)
        return yy
    def refine(self, variance=0.0):
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
        super().__init__()
        self.time_d = time_d
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.width = width
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.randn(time_d, out_channels,in_channels, width, width))
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_channels))
        
    def forward(self, t, x):
        # Use the trick where it's the same as index selection
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        #t_idx = torch.LongTensor([t_idx]).to(which_device(self))
        wij = self.weight[t_idx,:,:,:,:]
        bi = self.bias[t_idx,:]
        y = torch.nn.functional.conv2d(x, wij,bi, padding=self.padding)
        return y
    
    @torch.no_grad()
    def refine(self, variance=0.0):
        new = Conv2DODE(2*self.time_d,
                       self.in_channels,
                       self.out_channels,
                       width=self.width,
                       padding=self.padding).to(which_device(self))
        for t in range(self.time_d):
            new.weight.data[2*t:2*t+2,:,:,:,:] = self.weight.data[t,:,:,:,:]
        for t in range(self.time_d):
            new.bias.data[2*t:2*t+2,:] = self.bias.data[t,:]
        if variance != 0:
            new.weight.data *= 1.0 + variance * torch.randn_like(new.weight)
            new.bias.data *= 1.0 + variance * torch.randn_like(new.bias)
        return new

class Conv2DPolyODE(torch.nn.Module):
    def __init__(self, time_d, in_channels, out_channels,
                width=1, padding=1):
        super().__init__()
        self.time_d = time_d
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.width = width
        self.padding = padding
        self.weight = torch.nn.Parameter(torch.randn(time_d, out_channels,in_channels, width, width))
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_channels))
        
    def forward(self, t, x):
        """Shape function options: [1, t, t**2]
        [(1-t), (t)]"""
        wij = self.weight[0,:,:,:,:]*(1.0-t) + self.weight[1,:,:,:,:]*t
        bi = self.bias[0,:]*(1.0-t) + self.bias[1,:]*t
        y = torch.nn.functional.conv2d(x, wij,bi, padding=self.padding)
        return y
    
    @torch.no_grad()
    def refine(self, variance=0.0):
        """What does refine mean"""
        new = Conv2DPolyODE(
            self.time_d+1,
            self.in_channels,
            self.out_channels,
            width=self.width,
            padding=self.padding).to(which_device(self))
        # TODO this one is wrong
        new.weight.data[:self.time_d,:,:,:,:] = self.weight.data[t,:,:,:,:]
        new.bias.data[:self.time_d,:] = self.bias.data[t,:]
        if variance != 0:
            new.weight.data *= 1.0 + variance * torch.randn_like(new.weight)
            new.bias.data *= 1.0 + variance * torch.randn_like(new.bias)
        return new

class ShallowConv2DODE(torch.nn.Module):
    def __init__(self, time_d, in_features, hidden_features, 
                 width=3, padding=1,
                 act=torch.nn.functional.relu,
                 epsilon=1.0,
                 use_batch_norms="None",
                 use_skip_init=True,
                 shape_function='piecewise'):
        """
        Args:
        
        use_batch_norms: options are ("No"|False), "nn", "ode"
        """
        super().__init__()
        self.act = act
        self.epsilon = epsilon
        self.use_batch_norms = use_batch_norms
        self.use_skip_init = use_skip_init
        self.verbose=False
        self.shape_function = shape_function
        
        if shape_function == 'piecewise':
            self.L1 = Conv2DODE(time_d, in_features, hidden_features,
                            width=width, padding=padding)
            self.L2 = Conv2DODE(time_d, hidden_features, in_features,
                            width=width, padding=padding)
            if use_skip_init:
                self.skip_init = SkipInitODE(time_d)
        elif shape_function == 'poly':
            self.L1 = Conv2DPolyODE(time_d, in_features, hidden_features,
                            width=width, padding=padding)
            self.L2 = Conv2DPolyODE(time_d, hidden_features, in_features,
                            width=width, padding=padding)
            if use_skip_init:
                self.skip_init = SkipInitODE(1)

        if use_batch_norms=="nn":
            self.bn1 = nn.BatchNorm2d(
                hidden_features, affine=True, track_running_stats=True)
            self.bn2 = nn.BatchNorm2d(
                in_features, affine=True, track_running_stats=True)
        elif use_batch_norms=="ode":
            self.bn1 = BatchNorm2DODE(
                time_d, hidden_features, affine=True, track_running_stats=True)
            self.bn2 = BatchNorm2DODE(
                time_d, in_features, affine=True, track_running_stats=True)
        
    def forward(self, t, x):
        if self.verbose: print("shallow @ ",t)
        x = self.L1(t, x)
        x = self.act(x, inplace=True)
        if self.use_batch_norms=="nn":
            x = self.bn1(x)
        elif self.use_batch_norms=="ode":
            x = self.bn1(t,x)
        x = self.L2(t, x)
        x = self.act(x, inplace=True)
        if self.use_batch_norms=="nn":
            x = self.bn2(x)
        elif self.use_batch_norms=="ode":
            x = self.bn2(t,x)
        if self.use_skip_init:
            x = self.skip_init(t, x)
        return self.epsilon*x
    
    @torch.no_grad()
    def refine(self, variance=0.0):
        r_L1 = refine(self.L1, variance)
        r_L2 = refine(self.L2, variance)
        
        if self.use_skip_init:
            r_skip_init = refine(self.skip_init, variance)
        else:
            r_skip_init = None

        if self.use_batch_norms=="nn":
            self.bn1.track_running_stats = False
            self.bn2.track_running_stats = False
            r_bn1 = copy.deepcopy(self.bn1)
            r_bn2 = copy.deepcopy(self.bn2)
        elif self.use_batch_norms=="ode":
            r_bn1 = self.bn1.refine()
            r_bn2 = self.bn2.refine()
        else:
            r_bn1 = None
            r_bn2 = None

        # TODO Don't like it, it re-allocates the weights that we're gonna throw away
        new = copy.deepcopy(self) 
        new.L1 = r_L1
        new.L2 = r_L2
        new.skip_init = r_skip_init
        new.bn1 = r_bn1
        new.bn2 = r_bn2
        
        if self.use_batch_norms=="nn":
            self.bn1.track_running_stats = True
            self.bn2.track_running_stats = True

        return new

class ShallowConv2DODE_Flipped(ShallowConv2DODE):
    def forward(self, t, x):
        if self.verbose: print("shallow @ ",t)
        if self.use_batch_norms=="nn":
            x = self.bn1(x)
        elif self.use_batch_norms=="ode":
            x = self.bn1(t,x)
        x = self.act(x, inplace=True)
        x = self.L1(t, x)
        if self.use_batch_norms=="nn":
            x = self.bn2(x)
        elif self.use_batch_norms=="ode":
            x = self.bn2(t,x)
        x = self.act(x, inplace=True)
        x = self.L2(t, x)
        
        if self.use_skip_init:
            x = self.skip_init(t, x)
        return self.epsilon*x

class ODEify(torch.nn.Module):
    """Throws away the t."""
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x):
        return f(x)
    def refine(self):
        return ODEify(f)
    

class ODEBlock(torch.nn.Module):
    """Wraps an ode-model with the odesolve to fit into standard 
    models."""
    def __init__(self, net, n_time_steps=1, scheme='euler',
                 use_adjoint=False):
        super(ODEBlock,self).__init__()
        self.n_time_steps = n_time_steps
        self.scheme = scheme
        self.use_adjoint = use_adjoint
        # TODO: awk with piecewise constant centered on the half-cells
        self.ts = torch.linspace(0, 1.0, self.n_time_steps+1)
        self.net = net
        
    def forward(self,x):
        if self.use_adjoint:
            integ = torchdiffeq.odeint_adjoint
        else:
            integ = torchdiffeq.odeint
        h = integ(self.net, x, self.ts, method=self.scheme,
                  options=dict(enforce_openset=True)
                 )[-1,:,:]
        return h
    
    def refine(self, variance=0.0):
        r_net = refine(self.net, variance)
        new = ODEBlock(r_net, self.n_time_steps*2, scheme=self.scheme).to(which_device(self))
        return new
    
    def diffeq(self,x):
        hs = torchdiffeq.odeint(self.net, x, self.ts, method=self.scheme,
                                options=dict(enforce_openset=True)
                               )
        return hs
    
    def set_n_time_steps(self, n_time_steps):
        self.n_time_steps=n_time_steps
        self.ts = torch.linspace(0, 1.0, self.n_time_steps+1)

class ODEStitch(nn.Module):
    """Perfoms a downsampling stitch with the ResNet non-ode version.
    ODEs require in_features to be equal to out_features. This performs that one-time
    reshaping needing in spacial dimensions.
    """
    def __init__(self, in_features, out_features, hidden_features, 
                 width=3, padding=1,
                 act=torch.nn.functional.relu,
                 epsilon=1.0,
                 use_batch_norms=False,
                 use_skip_init=True,
                 stride=2):
        super().__init__()
        self.act = act
        self.epsilon = epsilon
        self.use_batch_norms = use_batch_norms
        self.use_skip_init = use_skip_init
        self.verbose=False
        self.downsample = nn.Conv2d(
                in_features, out_features, kernel_size=1, padding=0, stride=stride, bias=False)
        self.L1 = nn.Conv2d(
            in_features, hidden_features, stride=stride, kernel_size=width, padding=padding)
        self.L2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=width, padding=padding)
        if use_skip_init:
            self.skip_init = nn.Parameter(torch.zeros(1))
        if use_batch_norms:
            self.bn1 = torch.nn.BatchNorm2d(hidden_features)
            self.bn2 = torch.nn.BatchNorm2d(out_features)

    def forward(self, x):
        h = self.L1(x)
        h = self.act(h, inplace=True)
        if self.use_batch_norms:
            h = self.bn1(h)   
        h = self.L2(h)
        h = self.act(h, inplace=True)
        if self.use_batch_norms:
            h = self.bn2(h)           
        if self.use_skip_init:
            h = self.skip_init * h
        x_down = self.downsample(x)
        return x_down + h
    def refine(self, variance=0):
        return copy.deepcopy(self)


class BatchNorm2DODE(nn.Module):
    """A piece-wise chain of batch norms."""
    def __init__(self,
                 time_d,
                 features,
                 affine=True,
                 track_running_stats=True,
                 _force_bns=None):
        """Construct a new time-indexed batch norm group.
        
        _force_bns is meant to be called by refine, which overrides
        initialization.
        """
        super().__init__()
        self.time_d = time_d
        self.features = features
        if _force_bns is None:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(
                    features,
                    affine=affine,
                    track_running_stats=track_running_stats)
                for _ in range(time_d)
            ])
        else:
            self.bns = nn.ModuleList(_force_bns)

    def forward(self, t, x):
        tdx = piecewise_index(t, self.time_d)
        return self.bns[tdx](x)

    def refine(self, variance=0):
        _force_bns = []
        for my_bn in self.bns:
            for _ in range(2):
                _force_bns.append(copy.deepcopy(my_bn))
        new = BatchNorm2DODE(2*self.time_d, self.features, _force_bns=_force_bns)
        return new
