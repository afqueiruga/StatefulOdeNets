import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

# LeNetLike Model definition
class LeNetLike(nn.Module):
    def __init__(self, jump=0.0):
        super(LeNetLike, self).__init__()
        
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 90)
        self.fc3 = nn.Linear(90, 10)

        self.mode = 'normal'
        
    
    def change_mode(self, mode):
        assert(mode == 'normal' or mode == 'out_act')
        self.mode = mode

    def forward(self, x):
        bs = x.size(0)
        output_list = [deepcopy(x.data)]

        x = self.relu(F.max_pool2d(self.conv1(x), 2))
        output_list.append(deepcopy(x.data))
        
        x = self.relu(F.max_pool2d(self.conv2(x), 2))
        output_list.append(deepcopy(x.data))
        
        x = x.view(-1, 320)
        x = F.dropout(x, training=self.training) 
        
        x = self.relu(self.fc1(x))
        output_list.append(deepcopy(x.data))
        
        x = self.relu(self.fc2(x))
        output_list.append(deepcopy(x.data))
        
        x = self.fc3(x)

                
        output_list.append(deepcopy(x.data))
        
        
        
        if self.mode == 'normal':
            return x
        elif self.mode == 'out_act':
            output_list = [t.view(bs, -1) for t in output_list]
            return x, output_list
