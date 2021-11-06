#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:02:47 2021

@author: ben
"""
from continuous_net.tools.data_transform import DataTransform
from continuous_net.convergence import ConvergenceTester
import datasets
import glob
import os


DIR = "../stateful_cifar10/"

paths = glob.glob(f"{DIR}/*")


torch_train_data, torch_validation_data, torch_test_data = (datasets.get_dataset(name='CIFAR10', root='../'))
train_data = DataTransform(torch_train_data)
validation_data = DataTransform(torch_validation_data)
test_data = DataTransform(torch_test_data)



for path in paths:
    try:
        print(path)
        ct = ConvergenceTester(path)
        print(ct.eval_model)
        ct.perform_project_and_infer(test_data,  bases=["piecewise_constant"], n_bases=[16, 14, 12, 10, 8, 6, 4], schemes=['Euler'], n_steps=[16])


    except Exception as e:
            print("Error wih ", path, ": ", e)
