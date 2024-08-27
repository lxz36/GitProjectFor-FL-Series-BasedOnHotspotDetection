
#这个程序的主要目的是转化热区数据为npy 为联邦蒸馏程序创建数据集
import sys
import os
import random
import time
import numpy as np
from tqdm import trange
import configparser as cp
import copy
import argparse


if 'PYTHONPATH' in os.environ:
    # FIXME: unset this to make torchvision work in our server
    del os.environ['PYTHONPATH']

if 'OMP_DISPLAY_ENV' in os.environ:
    os.environ['OMP_DISPLAY_ENV'] = 'FALSE'


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from model_pytorch import Model
from data_pytorch_mixed import MixedLayoutHotspotDataset
from data_pytorch import alloc_benchmark
from tools import *


# arg_parser = argparse.ArgumentParser()
# arg_parser.add_argument("--model_path", type=str,
#                         default="./models/model")
#
# args = arg_parser.parse_args()


np.random.seed(42)


'''
Initialize Path and Global Params
'''
infile = cp.ConfigParser()
infile.read('iccad_config.ini')


# prepare testing set paths
test_data_ini_items = {
    'iccad2012': 'test_path_2012',
    'asml1':     'test_path_asml1',
    'asml2':     'test_path_asml2',
    'asml3':     'test_path_asml3',
    'asml4':     'test_path_asml4'}

test_data_keys_iccad2012 = ['iccad2012']
test_data_paths_iccad2012 = {
    ds_key: infile.get('dir', test_data_ini_items[ds_key])
    for ds_key in test_data_keys_iccad2012}

test_data_keys_asml1 = ['asml1']#, 'iccad2012'
test_data_paths_asml1 = {
    ds_key: infile.get('dir', test_data_ini_items[ds_key])
    for ds_key in test_data_keys_asml1}

'''
Define dataset preprocessing pipeline
'''
# load mean & std for normalization
normalization_dataset = {
    'iccad2012': (np.load('npy/iccad2012-mean.npy'),
                  np.load('npy/iccad2012-std.npy')),
    'asml1': (np.load('npy/asml1-mean.npy'),
              np.load('npy/asml1-std.npy')),
    'asml2': (np.load('npy/asml2-mean.npy'),
              np.load('npy/asml2-std.npy')),
    'asml3': (np.load('npy/asml3-mean.npy'),
              np.load('npy/asml3-std.npy')),
    'asml4': (np.load('npy/asml4-mean.npy'),
              np.load('npy/asml4-std.npy'))}


# train data pipeline
train_data_iccad = MixedLayoutHotspotDataset(
    benchmark_paths=['benchmarks/iccad2012_train']
    )#, 'benchmarks/asml1_train'

train_data_asml1 = MixedLayoutHotspotDataset(
    benchmark_paths=['benchmarks/asml1_train']
    )#, 'benchmarks/asml1_train'

# test data pipeline
test_data_iccad2012 = []
for _key in test_data_keys_iccad2012 :
    _path = test_data_paths_iccad2012[_key]
    test_data_iccad2012 += alloc_benchmark(
        benchmark_dir=_path,
        clients_num=1,
        transform='test',
        normalize=normalization_dataset[_key])

# test data pipeline
test_data_asml1 = []
for _key in test_data_keys_asml1 :
    _path = test_data_paths_asml1[_key]
    test_data_asml1 += alloc_benchmark(
        benchmark_dir=_path,
        clients_num=1,
        transform='test',
        normalize=normalization_dataset[_key])

#改变形状和切片为28*28的函数
def ReshapeAndSlice(def_in):
    length = len(def_in)
    def_out= np.reshape(def_in, (length, -1))#改变形状
    def_out =def_out[:,0:28*28]#切片为28*28
    def_out = np.reshape(def_out, (length,28,28))
    return def_out

#标准化训练数据集和对应的测试集
def StandarizedDataset(def_in,def_in1,standarized=False):
    max_length=(np.max(def_in)-np.min(def_in))/2
    if def_in1==None:  #只考虑训练集
        X_train = def_in
        if standarized:  # 标准化X

            # mean_image = np.mean(X_train, axis=0)
            X_train -= np.min(def_in)
            X_train = (X_train / max_length)-1
        return X_train
    X_train=def_in
    X_test = def_in1
    if standarized:  # 标准化X

        # mean_image = np.mean(X_train, axis=0)
        X_train -= np.min(def_in)
        X_test -= np.min(def_in)
        X_train =  (X_train / max_length)-1
        X_test = (X_test / max_length)-1
    return X_train,  X_test




#公共数据集为iccad
X_train_MNIST, y_train_MNIST=train_data_iccad.features,train_data_iccad.labels
# #改变形状和切片
# X_train_MNIST=ReshapeAndSlice(X_train_MNIST)
# ##标准化训练数据集
# X_train_MNIST=StandarizedDataset(X_train_MNIST,def_in1=None,standarized=True)
# public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}#publi


#私有数据集为iccad和asml1 测试集为iccad测试集+asml1测试集
#私有训练数据集为iccad
X_train_EMNIST_iccad, y_train_EMNIST_iccad=X_train_MNIST,y_train_MNIST#已经标准化了

#私有训练数据集为asml1
X_train_EMNIST_asml1, y_train_EMNIST_asml1=train_data_asml1.features,train_data_asml1.labels
# #改变形状和切片
# X_train_EMNIST_asml1=ReshapeAndSlice(X_train_EMNIST_asml1)
# ##标准化
# X_train_EMNIST_asml1=StandarizedDataset(X_train_EMNIST_asml1,def_in1=None,standarized=True)


#私有测试数据集为iccad
X_test_EMNIST_iccad, y_test_EMNIST_iccad=test_data_iccad2012[0].feature_buffer,test_data_iccad2012[0].label_buffer
#改变形状和切片
X_test_EMNIST_iccad=ReshapeAndSlice(X_test_EMNIST_iccad)
##标准化
X_test_EMNIST_iccad=StandarizedDataset(X_test_EMNIST_iccad,def_in1=None,standarized=True)


#私有测试数据集为asml1
X_test_EMNIST_asml1, y_test_EMNIST_asml1=test_data_asml1[0].feature_buffer,test_data_asml1[0].label_buffer
#改变形状和切片
X_test_EMNIST_asml1=ReshapeAndSlice(X_test_EMNIST_asml1)
##标准化
X_test_EMNIST_asml1=StandarizedDataset(X_test_EMNIST_asml1,def_in1=None,standarized=True)


# 存储公共数据集和iccad训练集
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\X_train_MNIST.npy", arr=X_train_MNIST)
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\y_train_MNIST.npy", arr=y_train_MNIST)

# 存储iccad测试集
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\X_test_EMNIST_iccad.npy", arr=X_test_EMNIST_iccad)
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\y_test_EMNIST_iccad.npy", arr=y_test_EMNIST_iccad)
# 存储asml1训练集
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\X_train_EMNIST_asml1.npy", arr=X_train_EMNIST_asml1)
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\y_train_EMNIST_asml1.npy", arr=y_train_EMNIST_asml1)
# 存储asml1测试集
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\X_test_EMNIST_asml1.npy", arr=X_test_EMNIST_asml1)
np.save(file="D:\SoftwareInstallation\Pycharm(学习资料)\\federated-learning-public-code-master\FedMD-master_xiugai_To_hotspot1\y_test_EMNIST_asml1.npy", arr=y_test_EMNIST_asml1)




#
# lin=test_data_asml1[0].feature_buffer
#
# interface
# print(lin)












