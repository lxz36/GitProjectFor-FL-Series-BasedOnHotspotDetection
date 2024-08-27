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
    'asml2': (np.load('npy/asml2-mean.npy'),
              np.load('npy/asml2-std.npy')),
    'asml3': (np.load('npy/asml3-mean.npy'),
              np.load('npy/asml3-std.npy')),
    'asml4': (np.load('npy/asml4-mean.npy'),
              np.load('npy/asml4-std.npy'))}

# train data pipeline
train_data = MixedLayoutHotspotDataset(
    benchmark_paths=['benchmarks/iccad2012_train',
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

'''
#Start the training
'''

criterion = soft_cross_entropy


server_optimizer = optim.Adam(
    server_model.parameters(),
    lr=lr_init,
    betas=(.9, .999))
server_lr_scheduler = optim.lr_scheduler.MultiStepLR(
    server_optimizer,
    milestones=[10, 20, 30, 40],
    gamma=0.5,
    verbose=False)


for server_round in range(max_round):
    print('\nserver round {}/{}'.format(server_round, max_round))

    server_model.train()

            
    '''
    Train model for one round (when reaches max iteration steps)
    '''
    print('Local layers training...')
    train(model=server_model,
          data_loader=train_loader,
          optimizer=server_optimizer,
          criterion=criterion,
          batch_size=train_batchsize,
          l2_reg_factor=weight_decay,
          group_lasso_strength=group_lasso_strength,
          max_itr=max_itr,
          prev_steps=server_steps,
          data_size=len(train_data),
          display_step=display_step,
          device=device,
          untrained_modules=[])
    # update steps
    server_steps += max_itr
    server_lr_scheduler.step()
    # save model
    torch.save(server_model.state_dict(),
               os.path.join(server_model_path, 'server-round{0:02d}.pt'.format(server_round)))

    '''
    Test model
    '''
    for i, test_data_key in enumerate(test_data_keys):
        print("\nTesting model on {}".format(test_data_key))
        test(model=server_model,
             data_loader=test_loader[i],
             criterion=criterion,
             batch_size=test_batchsize,
             display_step=display_step,
             device=device)
