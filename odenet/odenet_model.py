from torch import nn
import copy
import torch
#import torch.nn as nn
#import torch.nn.functional as F
import torchdiffeq
import math

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
        self.L1 = LinearODE(time_d, in_features, hidden_features)
        self.L2 = LinearODE(time_d, hidden_features, in_features)
    def forward(self, t, x):
        h = self.L1(t, x)
        hh = self.act(h)
        y = self.L2(t, hh)
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
    def __init__(self, time_d, in_channels, out_channels, width=1, padding=1):
        super(Conv2DODE,self).__init__()
        self.time_d = time_d
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.width = width
        self.padding = padding
        #self.weight = torch.nn.Parameter(torch.randn(time_d, out_channels, in_channels, width , width) / (out_channels)**0.5)
        self.weight = torch.nn.Parameter(torch.randn(time_d, out_channels, in_channels, width , width)) #* math.sqrt(2./(width*width*out_channels)))
        self.bias = torch.nn.Parameter(torch.zeros(time_d, out_channels))
        
            
    def forward(self, t, x):
        # Use the trick where it's the same as index selection
        t_idx = int(t*self.time_d)
        if t_idx==self.time_d: t_idx = self.time_d-1
        #t_idx = torch.LongTensor([t_idx]).to(which_device(self))
        wij = self.weight[t_idx,:,:,:,:]
        bi = self.bias[t_idx,:]
        y = torch.nn.functional.conv2d(x, wij, bi, padding=self.padding)
        return y
    
    def refine(self):
        new = Conv2DODE(2*self.time_d,
                       self.in_channels,
                       self.out_channels,
                       width=self.width,
                       padding=self.padding).to(which_device(self))
        for t in range(self.time_d):
            new.weight.data[2*t:2*t+2,:,:,:,:] = self.weight.data[t,:,:,:,:]
        for t in range(self.time_d):
            new.bias.data[2*t:2*t+2,:] = self.bias.data[t,:]
        return new






 
class ShallowConv2DODE(torch.nn.Module):
    def __init__(self, time_d, in_features, hidden_features, 
                 width=3, padding=1,
                 act=torch.nn.functional.relu,
                 epsilon=1.0,
                 use_batch_norms=True):
        super().__init__()
        self.act = act
        self.epsilon = epsilon
        self.use_batch_norms = use_batch_norms
        self.verbose=False
        
        self.L1 = Conv2DODE(time_d,in_features,hidden_features,
                            width=width, padding=padding)
        
        self.L2 = Conv2DODE(time_d,hidden_features,in_features,
                            width=width, padding=padding)
        
        self.SkipInit = nn.Parameter(torch.tensor(0.0).float())           

        
        if use_batch_norms:
            self.bn1 = torch.nn.BatchNorm2d(hidden_features)
            self.bn2 = torch.nn.BatchNorm2d(in_features)
            
            
        
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
            
            
        #x = self.SkipInit * x      
        return self.epsilon*x
    
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
            
            if self.use_batch_norms:
                self.bn1.track_running_stats = True
                self.bn2.track_running_stats = True
            return new



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
        #self.ts = torch.linspace(0, 1.0/self.n_time_steps, self.n_time_steps+1)
        #self.ts = torch.linspace(0, 1.0/self.n_time_steps, 1+1)
        self.ts = torch.linspace(0, 1.0, 1+1)

        self.net = net
        
    def forward(self,x):
        if self.use_adjoint:
            integ = torchdiffeq.odeint_adjoint
        else:
            integ = torchdiffeq.odeint
        h = integ(self.net, x, self.ts, method=self.scheme, options=dict(enforce_openset=True))[-1,:,:]
        return h
    
    def refine(self):
        newnet = self.net.refine()
        new = ODEBlock(newnet,self.n_time_steps*2,scheme=self.scheme).to(which_device(self))
        return new
    
    def diffeq(self,x):
        hs = torchdiffeq.odeint(self.net, x, self.ts, method=self.scheme,
                              options=dict(enforce_openset=True))
        return hs
    
#    def set_n_time_steps(self, n_time_steps):
#        self.n_time_steps=n_time_steps
#        self.ts = torch.linspace(0, 1.0, self.n_time_steps+1)



class ODEResNet_SingleSegment(nn.Module):
    """Uses one OdeBlock per segment."""
    def __init__(self,
                 ALPHA=16,
                 scheme='euler',
                 time_d=1,
                 in_channels=3,
                 use_batch_norms=True,
                 time_epsilon=1.0,
                 n_time_steps_per=1,
                 use_adjoint=False):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        self.use_batch_norms = use_batch_norms
        # This macro lets us make 3 of them concisely without typos
        _macro = lambda _alpha : \
            ODEBlock(
                ShallowConv2DODE(
                    time_d,
                    _alpha,
                    _alpha,
                    width=3,
                    epsilon=time_epsilon,
                    use_batch_norms=use_batch_norms),
                n_time_steps=time_d*n_time_steps_per,
                scheme=scheme,
                use_adjoint=use_adjoint)
        
        # The full resnet, with three segments of the above macro
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ALPHA, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(ALPHA),
            nn.ReLU(),
            _macro(ALPHA),
            
            nn.Conv2d(ALPHA, 2*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(2*ALPHA),
            _macro(2*ALPHA),
            
            nn.Conv2d(2*ALPHA, 4*ALPHA, kernel_size=1, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(4*ALPHA),
            _macro(4*ALPHA),
            
            #nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(4*ALPHA,10),
        )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                print(m.kernel_size[0])
                print('done')

            elif isinstance(m, Conv2DODE):
                n = m.width * m.width * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                print('Conv2DODE') 

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  
        
        
        
    def forward(self,x):
        return self.net(x)
    
    def refine(self):
        new = ODEResNet.__new__(ODEResNet)
        new.time_d = 2*self.time_d
        new.method = self.method
        new.net = nn.Sequential(*[ refine(mod) for mod in self.net])
        return new
