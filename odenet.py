import torch
import torchdiffeq

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
        return  x @ wij+ bi
    def refine(self):
        new = LinearODE(2*self.time_d,
                       self.in_features,
                       self.out_features)
        for t in range(self.time_d):
            new.weights[2*t:2*t+2,:,:] = self.weights[t,:,:]
        for t in range(self.time_d):
            new.bias[2*t:2*t+2,:] = self.bias[t,:]
        return new

class Conv2DODE(torch.nn.Module):
    pass

class ODEBlock(torch.nn.Module):
    """Wraps an ode-model with the odesolve to fit into standard 
    models."""
    def __init__(self,net,t_max=1.0,method='euler'):
        super(ODEBlock,self).__init__()
        self.t_max = t_max
        self.method = method
        self.ts = torch.tensor([0,t_max])
        self.net = net
    def forward(self,x):
        h = torchdiffeq.odeint(self.net, x, self.ts,
                               method=self.method)[1,:,:]
        #print(h.shape)
        return h
    
