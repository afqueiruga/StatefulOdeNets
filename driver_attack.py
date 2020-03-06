import numpy as np
import torch
import torch.nn as nn

import torch.nn.init as init

import time

from matplotlib import pylab as plt
from odenet import datasets
from odenet.odenet_cifar10 import ODEResNet
from odenet import refine_train

import importlib
importlib.reload(refine_train)
from odenet.helper import set_seed, get_device, which_device
#importlib.reload(odenet_cifar10)

import argparse

importlib.reload(refine_train)
from odenet.helper import set_seed, get_device, which_device


from attack_method import *
from advfuns import *
import pandas as pd
from prettytable import PrettyTable


from odenet.odenet_cifar10 import ODEResNet
from odenet import refine_train

for m in ['euler', 'rk4']:
    path = 'results/resnet_' + m + '.pkl'
    model = torch.load(path)[-1].eval()
    
    
    
    set_seed(1)
    device = get_device()
    
    
    
    from utils import *
    dataset = 'cifar10'
    batchSize = 200
    _, testloader = getData(name=dataset, train_bs=batchSize, test_bs=batchSize)
    
    
    
    
    #================================================
    # parameters
    #================================================
    
    runs = 1
    eps = 0.05
    iters = 1
    iter_df = 1
    
    jump = 0
    
    
    #================================================
    # begin simulation
    #================================================
    accuracy = pd.DataFrame()
    relative = pd.DataFrame()
    absolute = pd.DataFrame()
    attack_time = pd.DataFrame()
    
        
        
    #==============================================================================
    # Begin attack
    #==============================================================================
    for irun in range(runs):
            if dataset == 'mnist':
                num_data = 10000
                num_class = 9
    
                X_ori = torch.Tensor(num_data, 1, 28, 28)
                X_fgsm = torch.Tensor(num_data, 1, 28, 28)
                X_deepfool1 = torch.Tensor(num_data, 1, 28, 28)
                X_deepfool2 = torch.Tensor(num_data, 1, 28, 28)            
                X_tr = torch.Tensor(num_data, 1, 28, 28)            
                
            elif dataset == 'cifar10':
                num_data = 10000
                num_class = 9
                
                X_ori = torch.Tensor(num_data, 3, 32, 32)
                X_fgsm = torch.Tensor(num_data, 3, 32, 32)
                X_deepfool1 = torch.Tensor(num_data, 3, 32, 32)
                X_deepfool2 = torch.Tensor(num_data, 3, 32, 32)            
                print('cifar10')
            
            iter_fgsm = 0.
            iter_dp1 = 0.
            iter_dp2 = 0.
            iter_tr = 0.
            
            
            Y_test = torch.LongTensor(num_data)
            
            
            
            
            
            print('Run IFGSM')
            stat_time = time.time()
            for i, (data, target) in enumerate(testloader):
            
                X_ori[i*batchSize:(i+1)*batchSize, :] = data
                Y_test[i*batchSize:(i+1)*batchSize] = target
                
                
                X_fgsm[i*batchSize:(i+1)*batchSize,:], a = fgsm_adaptive_iter(model, data, target, eps, iterations=iters)
                #iter_fgsm += a
            #print('iters: ', iter_fgsm)
            time_ifgsm = time.time() - stat_time
            print('total_time: ', time_ifgsm)
                
            
            print('Run DeepFool (inf norm)')
            stat_time = time.time()
            for i, (data, target) in enumerate(testloader):
                X_deepfool1[i*batchSize:(i+1)*batchSize,:], a = deep_fool_iter(model, data, target, c=num_class, p=1, iterations=iter_df)
                iter_dp1 += a
            print('iters: ', iter_dp1)
            time_deepfool_inf = time.time() - stat_time
            print('total_time: ', time_deepfool_inf)
            
            
            print('Run DeepFool (two norm)')
            stat_time = time.time()
            for i, (data, target) in enumerate(testloader):
                X_deepfool2[i*batchSize:(i+1)*batchSize,:], a = deep_fool_iter(model, data, target, c=num_class, p=2, iterations=iter_df)
                iter_dp2 += a
            print('iters: ', iter_dp2)
            time_deepfool_two = time.time() - stat_time        
            print('total_time: ', time_deepfool_two)
            
    
    
            result_acc = np.zeros(9)
            result_ent = np.zeros(9)
            result_dis = np.zeros(9)
            result_dis_abs = np.zeros(9)
            result_large = np.zeros(9)
            
            
            result_acc[0], result_ent[0] = test_ori(model, testloader, num_data)
            result_acc[1], result_ent[1] = test_adv(X_fgsm, Y_test, model, num_data)
            result_acc[2], result_ent[2] = test_adv(X_deepfool1, Y_test, model, num_data)
            result_acc[3], result_ent[3] = test_adv(X_deepfool2, Y_test, model, num_data)
            
            
            # FGSM inf norm
            result_dis[1], result_dis_abs[1],  result_large[1]= distance(X_fgsm, X_ori, norm=1)
            
            # FGSM two norm
            result_dis[2], result_dis_abs[2],  result_large[2]= distance(X_fgsm, X_ori, norm=2)
            
            # Deepfool (inf) inf norm
            result_dis[3], result_dis_abs[3],  result_large[3]= distance(X_deepfool1, X_ori, norm=1)
    
            # Deepfool (inf) two norm
            result_dis[4], result_dis_abs[4],  result_large[4]= distance(X_deepfool1, X_ori, norm=2)
                    
            # Deepfool (two) inf norm
            result_dis[5], result_dis_abs[5],  result_large[5]= distance(X_deepfool2, X_ori, norm=1)
            
            # Deepfool (two) two norm
            result_dis[6], result_dis_abs[6],  result_large[6]= distance(X_deepfool2, X_ori, norm=2)        
            
            
            #***********************
            # Print results
            #***********************
            print('Jump value: ', jump)
            x = PrettyTable()
            x.field_names = [" ", "Clean Data", "IFGSM", "DeepFool_inf", "DeepFool", "TR"]
            x.add_row(np.hstack(('Accuracy: ',   np.round(result_acc[([0,1,2,3,4])], 5))))
            x.add_row(np.hstack(('Rel. Noise: ', np.round(result_dis[([0,1,3,6,8])], 5))))
            x.add_row(np.hstack(('Abs. Noise: ', np.round(result_dis_abs[([0,1,3,6,8])], 5))))
            print(x)
            
            
            #***********************
            # Add to pandas df
            #***********************
            
            
            s = pd.Series({"jump" : jump, 
                           "clean": np.round(result_acc[0], 3) , 
                           "IFGSM_inf": np.round(result_acc[1], 3),
                           "IFGSM_two": np.round(result_acc[1], 3), 
    
                           "DeepFool_max_inf": np.round(result_acc[2], 3), 
                           "DeepFool_max_two": np.round(result_acc[2], 3),
    
                           "DeepFool_inf": np.round(result_acc[3], 3), 
                           "DeepFool_two": np.round(result_acc[3], 3)                       
                           })    
            accuracy = accuracy.append(s, ignore_index=True)    
            
            
            s = pd.Series({"jump" : jump, 
                           "clean": np.round(result_dis[0], 3) , 
                           "IFGSM_inf": np.round(result_dis[1], 3),
                           "IFGSM_two": np.round(result_dis[2], 3), 
    
                           "DeepFool_max_inf": np.round(result_dis[3], 3), 
                           "DeepFool_max_two": np.round(result_dis[4], 3),
    
                           "DeepFool_inf": np.round(result_dis[5], 3), 
                           "DeepFool_two": np.round(result_dis[6], 3)                       
                           }) 
            relative = relative.append(s, ignore_index=True)    
                
                
            s = pd.Series({"jump" : jump, 
                           "clean": np.round(result_dis_abs[0], 3) , 
                           "IFGSM_inf": np.round(result_dis_abs[1], 3),
                           "IFGSM_two": np.round(result_dis_abs[2], 3), 
    
                           "DeepFool_max_inf": np.round(result_dis_abs[3], 3), 
                           "DeepFool_max_two": np.round(result_dis_abs[4], 3),
    
                           "DeepFool_inf": np.round(result_dis_abs[5], 3), 
                           "DeepFool_two": np.round(result_dis_abs[6], 3)                       
                           }) 
            absolute = absolute.append(s, ignore_index=True)    
    
    
            s = pd.Series({"jump": jump, 
                           "IFGSM": time_ifgsm, 
                           "DeepFool_inf": time_deepfool_inf, 
                           "DeepFool_two": time_deepfool_two})
            attack_time = attack_time.append(s, ignore_index=True)  