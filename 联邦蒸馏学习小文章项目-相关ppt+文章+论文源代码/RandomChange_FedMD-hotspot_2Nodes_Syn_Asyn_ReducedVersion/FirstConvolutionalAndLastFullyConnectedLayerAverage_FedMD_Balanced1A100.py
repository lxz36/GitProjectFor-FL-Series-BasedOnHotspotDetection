
#在源代码上面改动，改为数字和字母的二分类 数字和字母二分类：
import os
import errno
import argparse
import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from data_utils import load_MNIST_data, load_ConvConsistFedMD_data, generate_bal_private_data,\
generate_partial_data
from FirstConvolutionalAndLastFullyConnectedLayerAverage_FedMD import FirstConvolutionalAndLastFullyConnectedLayerAverage_FedMD
import time

def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')  #FedMD，一个联邦学习框架。参与者正在合作培训。
    parser.add_argument('-conf', metavar='conf_file', nargs=1, 
                        help='the config file for FedMD.'
                       )#用于训练的配置文件，默认的 conf_file 是 ./conf/ConvConsistFedMD_balance_conf.json, \

    conf_file = os.path.abspath("conf/ConvConsistFedMD_balance_conf.json")
    
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


if __name__ == "__main__":
    conf_file =  parseArg()
    with open(conf_file, "r", encoding='utf-8') as f:
        conf_dict = eval(f.read())


        N_parties = conf_dict["N_parties"]#N_parties = {int} 10
        private_classes = conf_dict["private_classes"]#private_classes = {list: 2} [0,1]
        N_samples_per_class = conf_dict["N_samples_per_class"] #N_samples_per_class = {int} 3
        
        N_rounds = conf_dict["N_rounds"]#N_rounds = {int} 13 总的训练轮次
        N_alignment = conf_dict["N_alignment"]#N_alignment联合 = {int} 5000
        N_private_training_round = conf_dict["N_private_training_round"]#N_private_training_round = {int} 10
        private_training_batchsize = conf_dict["private_training_batchsize"]#private_training_batchsize = {int} 5
        N_logits_matching_round = conf_dict["N_logits_matching_round"] #N_logits_matching_round = {int} 1
        asynchronousRates = conf_dict["asynchronousRates"]  # asynchronousRate 异步更新率，一般为50%
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"] #{int} 128
        model_saved_dir = conf_dict["model_saved_dir"]# {str} './pretrained_from_ICCAD/' 储存10个模型
        
        result_save_dir = conf_dict["result_save_dir"]# {str} './ResultOutput/'

        print("系统配置信息为：",conf_dict["configuration_description"])
    
    del conf_dict, conf_file


    start_time1 = time.time()
    # 缩小到原来总体数据集的的 1/set_reduct_para
    set_reduct_para=5  #至少分2份数据出来 set_reduct_para>=2

    # 存储公共数据集和iccad训练集
    X_train_ICCAD=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/X_train_ICCAD.npy")
    y_train_ICCAD=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/y_train_ICCAD.npy")
    # X_train_ConvConsistFedMD_iccad=X_train_ICCAD
    # y_train_ConvConsistFedMD_iccad=y_train_ICCAD

    #iccad训练集
    X_train_ConvConsistFedMD_iccad = X_train_ICCAD[:X_train_ICCAD.shape[0] // set_reduct_para, :, :]
    y_train_ConvConsistFedMD_iccad = y_train_ICCAD[:y_train_ICCAD.shape[0] // set_reduct_para]

    # iccad公共数据
    #ICCADnumsEach表示每一份有多少数据
    ICCADnumsEach=X_train_ICCAD.shape[0] // set_reduct_para
    if(set_reduct_para>1):
        X_train_ICCAD = X_train_ICCAD[ICCADnumsEach : ICCADnumsEach*2, :, :]
        y_train_ICCAD = y_train_ICCAD[ICCADnumsEach : ICCADnumsEach*2]



    # 存储iccad测试集
    X_test_ConvConsistFedMD_iccad=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/X_test_ConvConsistFedMD_iccad.npy")
    y_test_ConvConsistFedMD_iccad=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/y_test_ConvConsistFedMD_iccad.npy")
    # 存储asml1训练集
    X_train_ConvConsistFedMD_asml1=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/X_train_ConvConsistFedMD_asml1.npy")
    y_train_ConvConsistFedMD_asml1=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/y_train_ConvConsistFedMD_asml1.npy")
    # 缩小到原来总体数据集的的 1/set_reduct_para
    X_train_ConvConsistFedMD_asml1=X_train_ConvConsistFedMD_asml1[:X_train_ConvConsistFedMD_asml1.shape[0] // set_reduct_para, :, :]
    y_train_ConvConsistFedMD_asml1=y_train_ConvConsistFedMD_asml1[:y_train_ConvConsistFedMD_asml1.shape[0] // set_reduct_para]

    # 存储asml1测试集
    X_test_ConvConsistFedMD_asml1=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/X_test_ConvConsistFedMD_asml1.npy")
    y_test_ConvConsistFedMD_asml1=np.load(
        file="../OriginalHotspotSmallDataset_2Nodes/y_test_ConvConsistFedMD_asml1.npy")

    public_dataset = {"X": X_train_ICCAD, "y": y_train_ICCAD}  # public_dataset = {dict: 2}

    #前一半的节点用于装载iccad数据
    private_data, total_private_data \
        = generate_bal_private_data(X_train_ConvConsistFedMD_iccad, y_train_ConvConsistFedMD_iccad,
                                    N_parties=N_parties//2,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=True)  # data_overlap = False数据不可重叠
    # 后一半的节点用于装载asml1数据
    private_data1, total_private_data1 \
        = generate_bal_private_data(X_train_ConvConsistFedMD_asml1, y_train_ConvConsistFedMD_asml1,
                                    N_parties=N_parties//2,
                                    classes_in_use=private_classes,
                                    N_samples_per_class=N_samples_per_class,
                                    data_overlap=True)  # data_overlap = False数据不可重叠
    private_data = private_data + private_data1

    #iccad和asml1的测试集做拼接
    # X_test_ConvConsistFedMD_iccad=np.concatenate([X_test_ConvConsistFedMD_iccad,X_test_ConvConsistFedMD_asml1])
    # y_test_ConvConsistFedMD_iccad=np.concatenate([y_test_ConvConsistFedMD_iccad,y_test_ConvConsistFedMD_asml1])

    X_tmp_iccad, y_tmp_iccad = generate_partial_data(X=X_test_ConvConsistFedMD_iccad,y=y_test_ConvConsistFedMD_iccad,
                                         class_in_use=private_classes, verbose=True)
    private_test_data_iccad = {"X": X_tmp_iccad, "y": y_tmp_iccad}  # X_tmp = {ndarray: (4800, 28, 28)}；y_tmp = {ndarray: (4800,)}
    del X_tmp_iccad, y_tmp_iccad

    X_tmp_asml1, y_tmp_asml1 = generate_partial_data(X=X_test_ConvConsistFedMD_asml1,y=y_test_ConvConsistFedMD_asml1,
                                         class_in_use=private_classes, verbose=True)
    private_test_data_asml1 = {"X": X_tmp_asml1, "y": y_tmp_asml1}  # X_tmp = {ndarray: (4800, 28, 28)}；y_tmp = {ndarray: (4800,)}
    del X_tmp_asml1, y_tmp_asml1
    end_time1 = time.time()
    print(f"加載數據集执行时间，6s左右: {end_time1 - start_time1:.6f} 秒")

    for asynchronousRate in asynchronousRates:

        if model_saved_dir is not None:

            parties = []

            dpathICCAD = os.path.abspath(model_saved_dir)+"/ICCAD_Model"
            model_names = os.listdir(dpathICCAD)
            print("按照顺序读取和训练模型，一定要对准model_names",model_names)#一定要对准model_names ['CNN_16_16_32_32_1SorAid_250_2_iccad1.h5', 'CNN_16_16_32_32_1SorAid_250_2_iccad1.h5']
            for name in model_names:
                tmp = None
                tmp = load_model(os.path.join(dpathICCAD ,name))
                parties.append(tmp)

            dpathAsml = os.path.abspath(model_saved_dir)+"/Asml_Model"
            model_names = os.listdir(dpathAsml)
            print("按照顺序读取和训练模型，一定要对准model_names",model_names)#一定要对准model_names ['CNN_16_16_32_32_1SorAid_250_2_iccad1.h5', 'CNN_16_16_32_32_1SorAid_250_2_iccad1.h5']
            for name in model_names:
                tmp = None
                tmp = load_model(os.path.join(dpathAsml ,name))
                parties.append(tmp)


        start_time2 = time.time()
        fedmd = FirstConvolutionalAndLastFullyConnectedLayerAverage_FedMD(parties,
                      public_dataset = public_dataset,
                      private_data = private_data,
                      total_private_data = total_private_data,
                      private_test_data = private_test_data_iccad,
                      private_test_data_asml1=private_test_data_asml1,
                      N_rounds = N_rounds,
                      N_alignment = N_alignment,
                      N_logits_matching_round = N_logits_matching_round,
                      logits_matching_batchsize = logits_matching_batchsize,
                      N_private_training_round = N_private_training_round,
                      private_training_batchsize = private_training_batchsize,
                      asynchronousRate=asynchronousRate)
        # 要统计执行时间的代码
        end_time2 = time.time()
        print(f"初始化fedmd时间，83s左右: {end_time2 - start_time2:.6f} 秒")

        start_time3 = time.time()
        initialization_result = fedmd.init_result
        pooled_train_result = fedmd.pooled_train_result
        print("初始化先注釋，加快訓練")
        # 要统计执行时间的代码
        end_time3 = time.time()
        print(f"初始化结果执行时间，0s左右: {end_time3 - start_time3:.6f} 秒")

        start_time4 = time.time()
        collaboration_performance = fedmd.collaborative_training()
        # 要统计执行时间的代码
        end_time4 = time.time()
        print(f"fedmd蒸馏执行时间，3轮45s左右: {end_time4 - start_time4:.6f} 秒")


        if result_save_dir is not None:
            save_dir_path = os.path.abspath(result_save_dir)
            #make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        # 将初始化结果保存到 'init_result.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'init_result.pkl'), 'wb') as f:
            # 使用 pickle 模块的 dump 方法将 initialization_result 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(initialization_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 将训练结果保存到 'pooled_train_result.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'pooled_train_result.pkl'),
                  'wb') as f:
            # 使用 pickle 模块的 dump 方法将 pooled_train_result 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(pooled_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 将协作性能结果保存到 'col_performance.pkl' 文件中
        with open(os.path.join(save_dir_path, "asynchronousRate-" + str(asynchronousRate) + 'col_performance.pkl'), 'wb') as f:
            # 使用 pickle 模块的 dump 方法将 collaboration_performance 变量保存到文件中
            # wb 模式表示以二进制写入模式打开文件
            # protocol=pickle.HIGHEST_PROTOCOL 表示使用 pickle 模块的最高协议版本进行序列化
            pickle.dump(collaboration_performance, f, protocol=pickle.HIGHEST_PROTOCOL)
