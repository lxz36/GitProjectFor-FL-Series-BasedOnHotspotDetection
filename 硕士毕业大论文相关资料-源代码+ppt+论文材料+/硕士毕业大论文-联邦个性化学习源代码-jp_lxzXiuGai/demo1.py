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


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_path", type=str,
                        default="./models/model")

args = arg_parser.parse_args()


np.random.seed(42)


'''
Initialize Path and Global Params
'''
infile = cp.ConfigParser()
infile.read('iccad_config.ini')
# read .ini file
model_path = args.model_path
server_model_path = os.path.join(model_path, 'server')

# prepare server model path
os.makedirs(server_model_path, exist_ok=True)

# prepare testing set paths
test_data_ini_items = {
    'iccad2012': 'test_path_2012',
    'asml1':     'test_path_asml1',
    'asml2':     'test_path_asml2',
    'asml3':     'test_path_asml3',
    'asml4':     'test_path_asml4'}
test_data_keys = ['asml1', 'iccad2012']
test_data_paths = {
    ds_key: infile.get('dir', test_data_ini_items[ds_key])
    for ds_key in test_data_keys}


'''
Hyperparameter settings
'''

max_round = 50  # max training round in server, used to be 50
max_itr = 500
train_batchsize = 64  # training batch size in clients
test_batchsize = 256  # testing batch size
lr_init = 1e-3
bias_step = 6400  # step interval to adjust bias
weight_decay = 1e-5 # L2 regularization strength
group_lasso_strength = 0.

# other settings
display_step = 50  # display step
n_features = 32


'''
Define dataset preprocessing pipeline
'''
# load mean & std for normalization
normalization_dataset = {
    'iccad2012': (np.load('npy/iccad2012-mean.npy'),
                  np.load('npy/iccad2012-std.npy')),
    'asml1': (np.load('npy/asml1-mean.npy'),
              np.load('npy/asml1-std.npy')),
}

# train data pipeline
train_data = MixedLayoutHotspotDataset(
    benchmark_paths=[
                     'benchmarks/asml1_train']
    )

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=train_batchsize,
    shuffle=True,
    num_workers=16
    )
    

# test data pipeline
test_data = []
for _key in test_data_keys:
    _path = test_data_paths[_key]
    test_data += alloc_benchmark(
        benchmark_dir=_path,
        clients_num=1,
        transform='test',
        normalize=normalization_dataset[_key])
test_loader = [
    torch.utils.data.DataLoader(
        _data,
        batch_size=test_batchsize,
        shuffle=False,
        num_workers=16)
    for _data in test_data]


'''
Get clinets & server models ready
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

server_model = Model(n_features=n_features).to(device)
server_steps = 0