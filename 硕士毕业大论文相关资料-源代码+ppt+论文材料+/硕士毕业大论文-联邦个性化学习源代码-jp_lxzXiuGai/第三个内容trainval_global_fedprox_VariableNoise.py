#fedprox的程序，已经运行成功
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
from data_pytorch import alloc_benchmark
from tools import *

#2/dict写入json
import json



arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--n_iccad2012", type=int, default=2)  #1、初始default=0  设置为2加快训练
arg_parser.add_argument("--n_asml1", type=int, default=2)       #2、初始default=0
arg_parser.add_argument("--sel_ratio", type=float, default=1.)  #客户端选择率
arg_parser.add_argument("--model_path", type=str,
                        default="./models/model")
arg_parser.add_argument("--benchmark_path", type=str,
                        default="./benchmarks")

args = arg_parser.parse_args()

print('#clients of iccad2012:', args.n_iccad2012)
print('#clients of asml1:', args.n_asml1)
print('select ratio:', args.sel_ratio)


np.random.seed(42)


'''
Initialize Path and Global Params
'''
clients_num_per_ds = {
    'iccad2012': args.n_iccad2012,
    'asml1':     args.n_asml1,
    'asml2':     0,
    'asml3':     0,
    'asml4':     0,
}
clients_num = sum(clients_num_per_ds.values()) # iccad-2012 & industry1
print('Total num clients:', clients_num)
ds_keys = sorted(list(clients_num_per_ds.keys()))

infile = cp.ConfigParser()
infile.read('iccad_config.ini')
# read .ini file
client_train_benchmark_path = args.benchmark_path  # benchmark dir for training
model_path = args.model_path
server_model_path = os.path.join(model_path, 'server')

# prepare server model path
os.makedirs(server_model_path, exist_ok=True)

# prepare client paths
client_model_paths = [
    os.path.join(model_path, '_'.join(['client', ds_key, str(i)]))
    for ds_key in ds_keys
    for i in range(clients_num_per_ds[ds_key])]
for _path in client_model_paths:
    os.makedirs(_path, exist_ok=True)
print('client model path: {}'.format(client_model_paths))

# prepare client benchmark paths
client_train_paths = {}
for ds_key in ds_keys:
    client_train_paths[ds_key] = os.path.join(
        client_train_benchmark_path, ds_key + '_train')
print('client benchmark path:', client_train_paths)

# prepare testing set paths
test_data_ini_items = {
    'iccad2012': 'test_path_2012',
    'asml1':     'test_path_asml1',
    'asml2':     'test_path_asml2',
    'asml3':     'test_path_asml3',
    'asml4':     'test_path_asml4'}
test_data_keys = sorted([
    ds_key for ds_key in ds_keys if clients_num_per_ds[ds_key] > 0])
test_data_paths = {
    ds_key: infile.get('dir', test_data_ini_items[ds_key])
    for ds_key in test_data_keys}


'''
Hyperparameter settings
'''
max_itr_ds = {
    'iccad2012': 800,
    'asml1': 800}
weight_decay_ds = {
    'iccad2012': 1e-5,
    'asml1': 1e-5}
fedprox_mu_ds = {
    'iccad2012': 1e-3,
    'asml1': 1e-3}

max_round = 50  # max training round in server, used to be 50  总的训练轮次t
max_itr = sum([[max_itr_ds[_ds]] * clients_num_per_ds[_ds]
               for _ds in ds_keys if _ds in max_itr_ds], [])
train_batchsize = 64  # training batch size in clients
test_batchsize = 256  # testing batch size
lr_init = 1e-3
bias_step = 6400  # step interval to adjust bias
select_ratio = args.sel_ratio  # ratio of selected clients in each round
# weight_decay = 5e-4 # L2 regularization strength
weight_decay = sum([[weight_decay_ds[_ds]] * clients_num_per_ds[_ds]
                    for _ds in ds_keys if _ds in weight_decay_ds], [])
fedprox_mu = sum([[fedprox_mu_ds[_ds]] * clients_num_per_ds[_ds]
                  for _ds in ds_keys if _ds in fedprox_mu_ds], [])

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
train_data = list()
for ds_key in ds_keys:
    train_data += alloc_benchmark(
        benchmark_dir=client_train_paths[ds_key],
        clients_num=clients_num_per_ds[ds_key],
        transform='train',
        normalize=normalization_dataset[ds_key])
train_loader = [
    torch.utils.data.DataLoader(
        train_data[i],
        batch_size=train_batchsize,
        shuffle=True,
        # sampler=train_sampler[i],
        num_workers=0)#3、源代码num_workers=16
    for i in range(clients_num)]

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
        num_workers=0)#4、源代码num_workers=16
    for _data in test_data]


#从这里开始噪声的选择：
noiseVariances=[0.1];  #几个噪声的方差值

#包含所有的实验指标结果
ResultOfICCAD=dict() #ICCAD最终的实验结果，最外层是不同的噪声方差，最里层的这5个数据分别是Avg test loss， acc,tpr,fpr, FA，有50轮的数据
ResultOfAsml=dict() #Asml最终的实验结果，最外层是不同的噪声方差，最里层的这5个数据分别是Avg test loss， acc,tpr,fpr, FA，有50轮的数据

accResultOfICCAD=dict() #ICCAD最终的实验结果，把精度单独提取出来
accResultOfAsml=dict() #Asml最终的实验结果，把精度单独提取出来


for noiseVariance in noiseVariances:
    print("此时选择的噪声方差为：",noiseVariance,"-----------------------")

    '''
    Get clinets & server models ready
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    server_model = Model(n_features=n_features).to(device)

    client_models = [copy.deepcopy(server_model).to(device) for i in range(clients_num)]
    client_states = [
        {'model': client_models[i].state_dict(),
         'n_steps': 0,
        #  'n_samples': len(train_data[i])}
         'n_samples': 1}
        for i in range(clients_num)]
    latest_ckpt = [os.path.join(_path, 'client_state-step{}.h5') for _path in client_model_paths]
    client_step = np.zeros(clients_num, dtype=np.int32)


    '''
    #Start the training
    '''

    criterion = soft_cross_entropy


    client_optimizers = [
        optim.Adam(client_models[i].parameters(),
                   lr=lr_init,
                   betas=(.9, .999))
        for i in range(clients_num)
    ]
    client_lr_schedulers = [
        optim.lr_scheduler.MultiStepLR(opt,
                                       milestones=[10, 20, 30, 40],
                                       gamma=0.5,
                                       verbose=False)
        for opt in client_optimizers]


    server_conv1_norms = []
    for server_round in range(max_round):
        print('\nserver round {}/{}'.format(server_round, max_round))

        selected_client_num = max(int(clients_num * select_ratio), 1)
        client_selected = random.sample(range(clients_num), selected_client_num)
        client_selected = np.sort(client_selected)

        print('\nclients selected: {}'.format(client_selected))
        for client_idx in client_selected:  # 对每一个client开启一个对话
            client_model_name = client_model_paths[client_idx].split('/')[-1]
            print('\nTrain client {}: {}'.format(client_idx, client_model_name))
            client_path = latest_ckpt[client_idx]

            client_train_loader = train_loader[client_idx]

            client_model = client_models[client_idx]
            client_model.train()
            client_opt = client_optimizers[client_idx]
            print('Restoring client from server.')
            restore_from_server(client_model, server_model)

            '''
            Train client for one round (when reaches max iteration steps)
            '''
            train_fedprox(model=client_model,
                server_model=server_model,
                data_loader=client_train_loader,
                optimizer=client_opt,
                criterion=criterion,
                batch_size=train_batchsize,
                l2_reg_factor=weight_decay[client_idx],
                fedprox_mu=fedprox_mu[client_idx],
                max_itr=max_itr[client_idx],
                prev_steps=client_step[client_idx],
                data_size=len(train_data[client_idx]),
                display_step=display_step,
                device=device,
                untrained_modules=[])
            # this client's training is over
            client_step[client_idx] += max_itr[client_idx]
            # save client
            client_states[client_idx] = {
                'model': client_model.state_dict(),
                'n_steps': client_step[client_idx],
                'n_samples': len(train_data[client_idx])
            }
            torch.save(client_states[client_idx], client_path.format(client_step[client_idx]))
            print('Client model saved at', client_path.format(client_step[client_idx]))
        # all clients in this round over
        for _lr_sch in client_lr_schedulers:
            _lr_sch.step()
        print('\nStart Fed Avg...')
        fed_avgWithNoise(client_states, server_model,[],True,0,noiseVariance*(0.1)*pow(0.1,server_round//10)) #关键语句
        torch.save(server_model.state_dict(),
                   os.path.join(server_model_path, 'server-round{0:02d}.pt'.format(server_round)))
        # show derived feature importance
        server_conv1_norms.append(get_group_norm(server_model.conv1_1[0]))
        np.save('group_norm.npy', np.stack(server_conv1_norms))
        '''
        Test server
        '''
        for i, test_data_key in enumerate(test_data_keys):
            print("\nTesting server on", test_data_key)
            outSum=test(model=server_model,
                 data_loader=test_loader[i],
                 criterion=criterion,
                 batch_size=test_batchsize,
                 display_step=display_step,
                 device=device)

            if(test_data_key=="iccad2012"):
                if("噪声方差为%f时:"%(noiseVariance) in ResultOfICCAD):
                    ResultOfICCAD["噪声方差为%f时:"%(noiseVariance)].append(outSum)
                    accResultOfICCAD["噪声方差为%f时的精度acc变化:" % (noiseVariance)].append(outSum[1])
                else:
                    ResultOfICCAD["噪声方差为%f时:" % (noiseVariance)]=[]
                    ResultOfICCAD["噪声方差为%f时:" % (noiseVariance)].append(outSum)

                    accResultOfICCAD["噪声方差为%f时的精度acc变化:" % (noiseVariance)] = []
                    accResultOfICCAD["噪声方差为%f时的精度acc变化:" % (noiseVariance)].append(outSum[1])


            if(test_data_key=="asml1"):
                if("噪声方差为%f时:"%(noiseVariance) in ResultOfAsml):
                    ResultOfAsml["噪声方差为%f时:"%(noiseVariance)].append(outSum)
                    accResultOfAsml["噪声方差为%f时的精度acc变化:" % (noiseVariance)].append(outSum[1])
                else:
                    ResultOfAsml["噪声方差为%f时:" % (noiseVariance)]=[]
                    ResultOfAsml["噪声方差为%f时:" % (noiseVariance)].append(outSum)

                    accResultOfAsml["噪声方差为%f时的精度acc变化:" % (noiseVariance)] = []
                    accResultOfAsml["噪声方差为%f时的精度acc变化:" % (noiseVariance)].append(outSum[1])

print("最终的噪声50轮实验结果：-------------------------")
print("ICCAD的实验结果：-------------------------")
print(ResultOfICCAD)
print("单独提取ICCAD的精度acc结果：-------------------------")
print(accResultOfICCAD)

print("Asml的实验结果：-------------------------")
print(ResultOfAsml)
print("单独提取Asml的精度acc结果：-------------------------")
print(accResultOfAsml)

dictObj_iccadAndIndustry = {
'ICCAD的实验结果':ResultOfICCAD,
'单独提取ICCAD的精度acc结果':accResultOfICCAD,
'Asml的实验结果':ResultOfAsml,
'单独提取Asml的精度acc结果':accResultOfAsml,

}
fileObject = open('第三个内容trainval_global_fedprox_VariableNoise.json', 'w', encoding="utf8")
json.dump(dictObj_iccadAndIndustry, fileObject, ensure_ascii=False)
fileObject.close()