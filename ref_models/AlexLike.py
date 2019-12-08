import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from copy import deepcopy
# AlexLike Model definition
class AlexLike(nn.Module):
    def __init__(self):
        super(AlexLike, self).__init__()
       
        self.relu = nn.ReLU(inplace = True)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        
        self.fc1 = nn.Linear(5*5*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        self.mode = 'normal'
    
    def change_mode(self, mode):
        assert(mode == 'normal' or mode == 'out_act')
        self.mode = mode

    def forward(self, x):
        bs = x.size(0)
        output_list = [deepcopy(x.data)]

        x = self.relu(self.conv1(x))
        output_list.append(deepcopy(x.data))
        x = F.max_pool2d(self.relu(self.conv2(x)), 2)
        output_list.append(deepcopy(x.data))
        
        x = self.relu(self.conv3(x))
        output_list.append(deepcopy(x.data))
        x = F.max_pool2d(self.relu(self.conv4(x)), 2)
        output_list.append(deepcopy(x.data))

        x = x.view(x.size(0), -1)

        x = F.dropout(x, training=self.training)        
        x = self.relu(self.fc1(x))
        output_list.append(deepcopy(x.data))
        #x = F.dropout(x, training=self.training)                
        x = self.relu(self.fc2(x))
        output_list.append(deepcopy(x.data))
        x = self.fc3(x)
        
        output_list.append(deepcopy(x.data))
        if self.mode == 'normal':
            return x
        elif self.mode == 'out_act':
            output_list = [t.view(bs, -1) for t in output_list]
            return x, output_list
    
    
