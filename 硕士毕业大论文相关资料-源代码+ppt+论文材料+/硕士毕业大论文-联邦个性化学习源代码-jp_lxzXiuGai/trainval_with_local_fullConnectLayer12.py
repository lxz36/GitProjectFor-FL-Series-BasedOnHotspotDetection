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

from model_pytorch import Model  #引入我们自己构建的神经网络
from data_pytorch import alloc_benchmark
from tools import *



numberAsml1=[2,5]
numberIccad=[2,5]
RatioSum=[1.0,0.5]
resultSum=[]
for i_number in range(len(numberAsml1)):
    nowNumberAsml1=numberAsml1[i_number]
    nowNumberIccad2012 = numberIccad[i_number]
    for i_Ratio in range(len(RatioSum)):
        if nowNumberAsml1==nowNumberAsml1 and nowNumberAsml1==1 and RatioSum[i_Ratio]==0.5:
            continue
        nowRatio=RatioSum[i_Ratio]


        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--n_iccad2012", type=int, default=nowNumberIccad2012,  #1、初始default=0
                                help='number of iccad2012 clients')
        arg_parser.add_argument("--n_asml1", type=int, default=nowNumberAsml1,      #2、初始default=0
                                help='number of asml1 clients')
        arg_parser.add_argument("--sel_ratio", type=float, default=nowRatio,
                                help='ratio of selected clients in each round')
        arg_parser.add_argument("--model_path", type=str,
                                default="./models/model")
        arg_parser.add_argument("--benchmark_path", type=str,
                                default="./benchmarks")
        arg_parser.add_argument("--top-k-channels", type=int, default=26, metavar='K', #3、初始default=32
                                help='number of selected top K channels, '
                                'K=32 means all channels are used.')

        args = arg_parser.parse_args()  #args: benchmark_path='./benchmarks', model_path='./models/model', n_asml1=5, n_iccad2012=5, sel_ratio=1.0, top_k_channels=26 设置基本参数的作用


        print('#clients of iccad2012:', args.n_iccad2012)
        print('#clients of asml1:', args.n_iccad2012)
        print('select ratio:', args.sel_ratio)
        resultSumTemp = {'#clients of iccad2012:':args.n_iccad2012,'#clients of asml1:':args.n_iccad2012,'select ratio:': args.sel_ratio}

        np.random.seed(42) #只能是一次有效,seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。


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
        clients_num = sum(clients_num_per_ds.values()) # iccad-2012 & industry1 总的节点数量
        print('Total num clients:', clients_num)
        ds_keys = sorted(list(clients_num_per_ds.keys()))#ds_keys = ['asml1', 'asml2', 'asml3', 'asml4', 'iccad2012']

        infile = cp.ConfigParser()
        infile.read('iccad_config.ini')
        # read .ini file
        client_train_benchmark_path = args.benchmark_path # benchmark dir for training   client_train_benchmark_path='./benchmarks'
        model_path = args.model_path #'./benchmarks'
        server_model_path = os.path.join(model_path, 'server') #'./models/model\\server'

        # prepare server model path
        os.makedirs(server_model_path, exist_ok=True)

        # prepare client paths
        client_model_paths = [     #['./models/model\\client_asml1_0', './models/model\\client_asml1_1', './models/model\\client_asml1_2', './models/model\\client_asml1_3', './models/model\\client_asml1_4', './models/model\\client_iccad2012_0', './models/model\\client_iccad2012_1', './models/model\\client_iccad2012_2', './models/model\\client_iccad2012_3', './models/model\\client_iccad2012_4']
            os.path.join(model_path, '_'.join(['client', ds_key, str(i)]))
            for ds_key in ds_keys
            for i in range(clients_num_per_ds[ds_key])]
        for _path in client_model_paths:
            os.makedirs(_path, exist_ok=True)   #创建该文件夹
        print('client model path: {}'.format(client_model_paths))

        # prepare client benchmark paths
        client_train_paths = {}
        for ds_key in ds_keys:
            client_train_paths[ds_key] = os.path.join(
                client_train_benchmark_path, ds_key + '_train')
        print('client benchmark path:', client_train_paths)#client_train_paths ={'asml1': './benchmarks\\asml1_train', 'asml2': './benchmarks\\asml2_train', 'asml3': './benchmarks\\asml3_train', 'asml4': './benchmarks\\asml4_train', 'iccad2012': './benchmarks\\iccad2012_train'}

        # prepare testing set paths
        test_data_ini_items = {
            'iccad2012': 'test_path_2012',
            'asml1':     'test_path_asml1',
            'asml2':     'test_path_asml2',
            'asml3':     'test_path_asml3',
            'asml4':     'test_path_asml4'}
        test_data_keys = sorted([  #test_data_keys = ['asml1', 'iccad2012']
            ds_key for ds_key in ds_keys if clients_num_per_ds[ds_key] > 0])
        test_data_paths = {  #test_data_paths = {'asml1': './wwy/benchmarks/asml1/test', 'iccad2012': './wwy/benchmarks/iccad_2012/test'}
            ds_key: infile.get('dir', test_data_ini_items[ds_key])
            for ds_key in test_data_keys}


        '''
        Hyperparameter settings
        '''
        max_itr_local_ds = {  #max_itr_local_ds = {'iccad2012': 500, 'asml1': 500}
            'iccad2012': 500,
            'asml1': 500}
        max_itr_global_ds = { #max_itr_global_ds ={'iccad2012': 1500, 'asml1': 1500}
            'iccad2012': 1500,
            'asml1': 1500}
        weight_decay_ds = {  #weight_decay_ds ={'iccad2012': 1e-05, 'asml1': 1e-05}
            'iccad2012': 1e-5,
            'asml1': 1e-5}

        max_round = 30  # max training round in server, used to be 50
        max_itr_local = sum([[max_itr_local_ds[_ds]] * clients_num_per_ds[_ds]  #max_itr_local =[500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
                            for _ds in ds_keys if _ds in max_itr_local_ds], [])
        max_itr_global = sum([[max_itr_global_ds[_ds]] * clients_num_per_ds[_ds] #max_itr_global = [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]
                             for _ds in ds_keys if _ds in max_itr_global_ds], [])
        train_batchsize = 64  # training batch size in clients
        test_batchsize = 256  # testing batch size
        lr_init = 1e-3
        bias_step = 6400  # step interval to adjust bias 调整偏差的步长间隔
        select_ratio = args.sel_ratio  # ratio of selected clients in each round
        # weight_decay = 5e-4 # L2 regularization strength
        weight_decay = sum([[weight_decay_ds[_ds]] * clients_num_per_ds[_ds]  #权重的衰变weight_decay =[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05]
                            for _ds in ds_keys if _ds in weight_decay_ds], [])
        group_lasso_strength = 0.

        # other settings
        display_step = 50  # display step显示步骤
        n_features = args.top_k_channels
        # sel_ch = [7, 18, 26, 13,  9, 31, 11,  5, 20, 16, 25, 24, 19, 22, 30, 28,  8, 21,  2, 17, 29,
        #           3, 27,  4, 12,  0] # group lasso 26
        sel_ch = [0, 12, 4, 27, 3, 29, 17, 2, 21, 8, 28, 30, 22, 19, 24, 25, 16, 20, 5, 11, 31, 9, 13, 26, 18, 7, 23, 15, 1, 14, 10, 6]
        sel_ch = sel_ch[:args.top_k_channels]

        '''
        Define dataset preprocessing pipeline
        '''
        # load mean & std for normalization
        normalization_dataset = {   #normalization_dataset =
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

        # train data pipeline训练数据管道
        train_data = list()  #存着10个训练数据集
        for ds_key in ds_keys:
            train_data += alloc_benchmark(
                benchmark_dir=client_train_paths[ds_key],
                clients_num=clients_num_per_ds[ds_key],
                transform='train',
                normalize=normalization_dataset[ds_key],
                sel_ch=sel_ch)

        train_loader = [
            torch.utils.data.DataLoader(
                train_data[i],
                batch_size=train_batchsize,
                shuffle=True,
                num_workers=0,    #4、源代码num_workers=16
                pin_memory=True)
            for i in range(clients_num)]

        # test data pipeline
        test_data = [] #存着10个测试数据集
        for _key in test_data_keys:
            _path = test_data_paths[_key]
            test_data += alloc_benchmark(
                benchmark_dir=_path,
                clients_num=1,
                transform='test',
                normalize=normalization_dataset[_key],
                sel_ch=sel_ch)
        test_loader = [
            torch.utils.data.DataLoader(
                _data,
                batch_size=test_batchsize,
                shuffle=False,
                num_workers=0,    #5、源代码num_workers=16
                pin_memory=True)
            for _data in test_data]


        '''
        Get clinets & server models ready
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', device)

        server_model = Model(n_features=n_features).to(device) #revise1这里需要修改一下

        client_models = [copy.deepcopy(server_model).to(device) for i in range(clients_num)]#revise2
        client_states = [      #revise3
            {'model': client_models[i].state_dict(),
             'n_steps': 0,
            #  'n_samples': len(train_data[i])}
             'n_samples': 1}
            for i in range(clients_num)]
        latest_ckpt = [os.path.join(_path, 'client_state-step{}.h5') for _path in client_model_paths]
        client_step = np.zeros(clients_num, dtype=np.int32) #client_step =[0 0 0 0 0 0 0 0 0 0]


        '''
        #Start the training
        '''

        criterion = soft_cross_entropy


        client_optimizers = [   #revise4 不一定改
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


        for server_round in range(max_round):
            print('\nserver round {}/{}'.format(server_round, max_round))

            selected_client_num = max(int(clients_num * select_ratio), 1)  #selected_client_num =10
            client_selected = random.sample(range(clients_num), selected_client_num) #client_selected =[0 1 2 3 4 5 6 7 8 9]
            client_selected = np.sort(client_selected)

            print('\nclients selected: {}'.format(client_selected))
            for client_idx in client_selected:  # 对每一个client开启一个对话
                client_model_name = client_model_paths[client_idx].split('/')[-1] #client_model_name ='model\\client_asml1_0'
                print('\nTrain client {}: {}'.format(client_idx, client_model_name))
                client_path = latest_ckpt[client_idx] #client_path ='./models/model\\client_asml1_0\\client_state-step{}.h5'

                client_train_loader = train_loader[client_idx] #加载第几个数据集

                client_model = client_models[client_idx]
                client_model.train()
                client_opt = client_optimizers[client_idx]

                '''
                Restore only global params.
                '''
                print('Restoring client from server.')
                restore_from_server(client_model, server_model, names_not_restore=[ 'fc1.0','final_fc' ]) #revise5 个性化层  对标靖宇的程序，当个性化层有第一个全连接层时，用fc1.0

                '''
                Train client for one round (when reaches max iteration steps)
                '''
                print('Local layers training...')
                train(model=client_model,    #revise6
                      data_loader=client_train_loader,
                      optimizer=client_opt,
                      criterion=criterion,
                      batch_size=train_batchsize,
                      l2_reg_factor=weight_decay[client_idx],
                      group_lasso_strength=group_lasso_strength,
                      max_itr=max_itr_local[client_idx],
                      prev_steps=client_step[client_idx],
                      data_size=len(train_data[client_idx]),
                      display_step=display_step,
                      device=device,
                      untrained_modules=['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'   #全局层
                         ])
                # update steps
                client_step[client_idx] += max_itr_local[client_idx]
                if server_round < max_round - 1 :
                    print('All layers training...') #全部层的训练
                    train(model=client_model,
                        data_loader=client_train_loader,
                        optimizer=client_opt,
                        criterion=criterion,
                        batch_size=train_batchsize,
                        l2_reg_factor=weight_decay[client_idx],
                        group_lasso_strength=group_lasso_strength,
                        max_itr=max_itr_global[client_idx],
                        prev_steps=client_step[client_idx],
                        data_size=len(train_data[client_idx]),
                        display_step=display_step,
                        device=device,
                        untrained_modules=[])
                    # this client's training is over
                    client_step[client_idx] += max_itr_global[client_idx]
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
            fed_avg(client_states, server_model, names_not_merge=['fc1.0', 'final_fc'])  #关键语句revise5.3全局层 names_not_merge=个性化层
            torch.save(server_model.state_dict(),
                       os.path.join(server_model_path, 'server-round{0:02d}.pt'.format(server_round)))

            '''
            Test clients
            '''
            asml1AccFprTprResult = [0, 0, 0, 0,
                                    0]  # 里面5个元素，分别是'Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'
            iccadAccFprTprResult=[0,0,0,0,0] #里面5个元素，分别是'Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'

            #这里是每一个模型对所有的测试集（asml1和Iccad2012）做精度测试
            # for i, test_data_key in enumerate(test_data_keys):
            #     for client_idx, client_model in enumerate(client_models):
            #         client_model_name = client_model_paths[client_idx].split('/')[-1]
            #         print("\nTesting {} on {}".format(client_model_name, test_data_key))
            #         outSum=test(model=client_model,   #outSum=[];里面5个元素，分别是'Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'
            #             data_loader=test_loader[i],
            #             criterion=criterion,
            #             batch_size=test_batchsize,
            #             display_step=display_step,
            #             device=device)
            #
            #         if client_model_name in ["client_asml1_0","client_asml1_1","client_asml1_2","client_asml1_3","client_asml1_4","client_asml1_5"] and test_data_key=="asml1":
            #             for i_Result in range(len(asml1AccFprTprResult)):
            #                 asml1AccFprTprResult[i_Result] += outSum[i_Result]/ (clients_num // 2)
            #         if client_model_name in ["client_iccad2012_0","client_iccad2012_1","client_iccad2012_2","client_iccad2012_3","client_iccad2012_4","client_iccad2012_5"] and test_data_key=="iccad2012":
            #             for i_Result in range(len(iccadAccFprTprResult)):
            #                 iccadAccFprTprResult[i_Result] += outSum[i_Result]/ (clients_num // 2)

            #这里是每一个模型对对应的测试集（asml1或者Iccad2012）做精度测试
            for i, test_data_key in enumerate(test_data_keys):
                for client_idx, client_model in enumerate(client_models):
                    client_model_name = client_model_paths[client_idx].split('/')[-1]

                    if  "asml1" in  client_model_name and test_data_key=="asml1":
                        print("\nTesting {} on {}".format(client_model_name, test_data_key))
                        outSum = test(model=client_model,
                                      # outSum=[];里面5个元素，分别是'Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'
                                      data_loader=test_loader[i],
                                      criterion=criterion,
                                      batch_size=test_batchsize,
                                      display_step=display_step,
                                      device=device)
                        for i_Result in range(len(asml1AccFprTprResult)):
                            asml1AccFprTprResult[i_Result] += outSum[i_Result]/ (nowNumberAsml1)
                    if  "iccad2012" in client_model_name and test_data_key=="iccad2012":
                        print("\nTesting {} on {}".format(client_model_name, test_data_key))
                        outSum = test(model=client_model,
                                      # outSum=[];里面5个元素，分别是'Avg test loss: {0:.5f}, Avg test acc: {1:.5f}, Avg tpr: {2:.5f}, Avg fpr: {3:.5f}, total FA: {4:d}'
                                      data_loader=test_loader[i],
                                      criterion=criterion,
                                      batch_size=test_batchsize,
                                      display_step=display_step,
                                      device=device)
                        for i_Result in range(len(iccadAccFprTprResult)):
                            iccadAccFprTprResult[i_Result] += outSum[i_Result]/ (nowNumberIccad2012)

            print("asml1AccFprTprResult:",asml1AccFprTprResult) #avgResult
            print("iccadAccFprTprResult:",iccadAccFprTprResult) #avgResult

        resultSumTemp["asml1AccFprTprResult:"]=tuple(asml1AccFprTprResult)
        resultSumTemp["iccadAccFprTprResult:"] = tuple(iccadAccFprTprResult)
        print("--------一种情况的结果：----------")
        print(resultSumTemp)
        resultSum.append(resultSumTemp)

print("最终结果：",resultSum)

