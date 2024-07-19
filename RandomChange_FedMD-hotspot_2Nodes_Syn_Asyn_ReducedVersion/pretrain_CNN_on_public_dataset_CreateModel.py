import os
import errno
import sys
import argparse
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from data_utils import load_MNIST_data # 从 data_utils 文件导入函数 load_MNIST_data
from Neural_Networks import  cnn_4ConvLayer_2fcLayer_model # 从 Neural_Networks 文件导入函数 cnn_2layer_fc_model 和 cnn_4ConvLayer_2fcLayer_model


def parseArg():
    # 解析命令行参数的函数
    parser = argparse.ArgumentParser(description='Train an array of Neural Networks on either ICCAD or CIFAR')  # 创建 ArgumentParser 对象，并添加描述
    parser.add_argument('-conf', metavar='conf_file', nargs=1,
                        help='the config file for training, \
                        for training on ICCAD, the default conf_file is ./conf/pretrain_MNIST.json, \
                        for training on CIFAR, the default conf_file is ./conf/pretrain_CIFAR.json.'
                       )  # 添加 -conf 参数，用于指定训练配置文件

    conf_file = os.path.abspath("conf/pretrain_ICCAD_conf.json")  # 设置默认的配置文件路径

    if len(sys.argv) > 1:  # 如果命令行参数长度大于1
        args = parser.parse_args(sys.argv[1:])  # 解析命令行参数
        if args.conf:  # 如果指定了 -conf 参数
            conf_file = args.conf[0]  # 使用指定的配置文件
    return conf_file  # 返回配置文件路径


def train_models(models, X_train, y_train, X_test_iccad, y_test_iccad, X_test_asml, y_test_asml,
                 is_show = False, save_dir = "./", save_names = None,
                 early_stopping = True,
                 min_delta = 0.001, patience = 3, batch_size = 128, epochs = 2, is_shuffle=True, verbose = 1,  # 设置训练参数，默认为提前停止，批次大小为128，训练轮数为2等
                 ):
    '''
    Train an array of models on the same dataset.
    We use early termination to speed up training.
    在同一数据集上训练一组模型。
    我们使用提前终止来加速训练。
    '''
    resulting_val_acc = []  # 存储验证集准确率的列表
    record_result = []  # 存储训练结果的列表

    for n, model in enumerate(models):  # 遍历所有模型
        print("Training model ", n)  # 打印正在训练的模型编号

        #不训练模型，产生初始模型
        # if early_stopping:  # 如果启用了提前停止
        #     model.fit(X_train, y_train,
        #               validation_data = (X_test, y_test),  # 验证数据集
        #               callbacks=[EarlyStopping(monitor='val_acc', min_delta=min_delta, patience=patience)],  # 设置提前停止的回调函数
        #               batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose  # 训练参数
        #              )
        # else:  # 如果没有启用提前停止
        #     model.fit(X_train, y_train,
        #               validation_data = (X_test, y_test),  # 验证数据集
        #               batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose  # 训练参数
        #              )
        # 记录最后一个 epoch 的验证集准确率
        # resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        # # 记录训练和验证的准确率和损失
        # record_result.append({"train_acc": model.history.history["accuracy"],
        #                       "val_acc": model.history.history["val_accuracy"],
        #                       "train_loss": model.history.history["loss"],
        #                       "val_loss": model.history.history["val_loss"]})

        # y_pred_iccad = model["model_classifier"].predict(X_test_iccad["X"], verbose=0).argmax(axis=1)
        # print("model_{0}".format(n),"在iccad测试集的准确率为：",np.mean(y_test_iccad["y"] == y_pred_iccad))
        # y_pred_asml = model["model_classifier"].predict(X_test_asml["X"], verbose=0).argmax(axis=1)
        # print("model_{0}".format(n), "在asml测试集的准确率为：", np.mean(y_test_asml["y"] == y_pred_asml))

        # 评估模型在测试集上的性能
        loss, accuracy = model.evaluate(X_test_iccad, y_test_iccad, verbose=1)
        print(f"Test ICCAD Loss: {loss}")
        print(f"Test ICCAD Accuracy: {accuracy}")

        # 评估模型在asml测试集上的性能
        loss, accuracy = model.evaluate(X_test_asml, y_test_asml, verbose=1)
        print(f"Test asml Loss: {loss}")
        print(f"Test asml Accuracy: {accuracy}")

        save_dir_path = os.path.abspath(save_dir)  # 获取保存目录的绝对路径
        # 创建目录
        try:
            os.makedirs(save_dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # 设置模型保存文件名
        if save_names is None:
            file_name = save_dir + "model_{0}".format(n) + ".h5"
        else:
            file_name = save_dir + save_names[n] + ".h5"
        print("当前模型是：",file_name)
        print("----------------------------------------------------------")
        model.save(file_name)  # 保存模型

    if is_show:  # 如果启用了显示
        print("pre-train accuracy: ")
        print(resulting_val_acc)  # 打印验证集准确率

    return record_result  # 返回记录的结果


models = {"cnn_4ConvLayer_2fcLayer_model": cnn_4ConvLayer_2fcLayer_model,    # 定义两种模型
          "cnn_4ConvLayer_2fcLayer_model": cnn_4ConvLayer_2fcLayer_model}


if __name__ == "__main__":
    conf_file =  parseArg()  # 获取配置文件路径
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())  # 读取并解析配置文件
    dataset = conf_dict["data_type"]  # 获取数据集类型
    n_classes = conf_dict["n_classes"]  # 获取类的数量
    model_config = conf_dict["models"]  # 获取模型配置
    train_params = conf_dict["train_params"]  # 获取训练参数
    save_dir = conf_dict["save_directory"]  # 获取保存目录
    save_names = conf_dict["save_names"]  # 获取保存文件名
    early_stopping = conf_dict["early_stopping"]  # 获取是否提前停止

    del conf_dict  # 删除配置字典

    if dataset == "ICCAD":
        input_shape = (144,32)  # 设置输入形状


        # 存储公共数据集和iccad训练集 是原来的1/10 用来产生初始模型
        # X_train_ICCAD = np.load(
        #     file="../OriginalHotspotSmallDataset_2Nodes/X_train_ICCAD.npy")
        # y_train_ICCAD = np.load(
        #     file="../OriginalHotspotSmallDataset_2Nodes/y_train_ICCAD.npy")
        #
        # # X_train_ConvConsistFedMD_iccad = X_train_ICCAD
        # # y_train_ConvConsistFedMD_iccad = y_train_ICCAD
        # # 存储iccad测试集
        # X_test_ConvConsistFedMD_iccad = np.load(
        #     file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_iccad.npy")
        # y_test_ConvConsistFedMD_iccad = np.load(
        #     file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_iccad.npy")

        # # 存储asml1训练集
        # X_train_ConvConsistFedMD_asml1 = np.load(
        #     file="..\OriginalHotspotDataset\X_train_ConvConsistFedMD_asml1.npy")
        # y_train_ConvConsistFedMD_asml1 = np.load(
        #     file="..\OriginalHotspotDataset\y_train_ConvConsistFedMD_asml1.npy")
        # # 存储asml1测试集
        # X_test_ConvConsistFedMD_asml1 = np.load(
        #     file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_asml1.npy")
        # y_test_ConvConsistFedMD_asml1 = np.load(
        #     file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_asml1.npy")

        X_test_iccad=np.load(
            file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_iccad.npy")
        y_test_iccad=np.load(
            file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_iccad.npy")
        X_test_asml=np.load(
            file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_asml1.npy")
        y_test_asml = np.load(
            file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_asml1.npy")
        X_train, y_train= None,None # 设置训练和测试数据集

    else:
        print("Unknown dataset. Program stopped.")  # 如果数据集类型未知，打印错误信息并退出程序
        sys.exit()

    pretrain_models = []  # 预训练的模型列表，目前为空
    for i, item in enumerate(model_config):  # 遍历所有模型配置
        name = item["model_type"]  # 获取模型类型
        model_params = item["params"]  # 获取模型参数
        tmp = models[name](n_classes=n_classes,
                           input_shape=input_shape,
                           **model_params)  # 创建模型实例

        print("打印模型编号和名称")
        print("model {0} : {1}".format(i, save_names[i]))  # 打印模型编号和名称
        print(tmp.summary())  # 打印模型结构
        pretrain_models.append(tmp)  # 将模型添加到预训练模型列表中

    # 训练模型
    record_result = train_models(pretrain_models, X_train, y_train, X_test_iccad, y_test_iccad, X_test_asml, y_test_asml,   # 传入预训练模型、训练数据和测试数据
                                 save_dir = save_dir, save_names = save_names, is_show=True,  # 传入保存目录和文件名
                                 early_stopping = early_stopping,  # 是否提前停止
                                 **train_params  # 传入训练参数
                                )

    with open('pretrain_result.pkl', 'wb') as f:  # 打开文件用于写入
        pickle.dump(record_result, f, protocol=pickle.HIGHEST_PROTOCOL)  # 将训练结果保存到文件中
    print('完成')  # 打印完成信息
