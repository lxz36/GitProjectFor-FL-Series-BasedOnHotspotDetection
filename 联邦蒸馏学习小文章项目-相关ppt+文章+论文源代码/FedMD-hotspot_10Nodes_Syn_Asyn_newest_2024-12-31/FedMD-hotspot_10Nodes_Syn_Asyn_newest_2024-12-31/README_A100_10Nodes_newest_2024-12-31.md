
零、程序相关

0.1、预训练相关文件
pretrain_CNN_on_public_dataset是预先验证的程序，产生预训练模型
pretrain_CNN_on_public_dataset.py【预训练主程序】
Neural_Networks.py【定义神经网络】
pretrain_ICCAD_conf.json【定义参数】
输出几个模型到pretrained_from_ICCAD文件夹

0.2、#目前整个工程是改造成为光刻热区的联邦蒸馏检测模型
FConvConsistFedMD_Balanced1 {(iccad对应6层CNN神经网络和 industry 对应7层CNN神经网络)}是**主程序**
运行前检查3要素 训练数据 输入的初始模型  程序 是否合理

目前fedMD-ConV已经全面实现了，调试参数看看效果如何，试试实现fed和fedProx性能比较


一、时间规划，思路顺序
1、先把卷积层加权平均加入训练（done）
2、改造为可以调整节点数量、数据集大小、同步异步的统一程序（需要注意的点，done）

2.2调整数据集大小相关：注意调整ConvConsistFedMD_balance_conf.json文件的N_samples_per_class参数（原始值为8000）和
"private_training_batchsize"参数（原始值为64）和"N_alignment"（原始值为30000）,

2.2.1、数据规模缩小10倍数，再分为10个节点，相当于蒸馏公共数据集是原来的1/10；10个节点，
每个节点保存的数据是原来的1/100 为iccad180  industry840， 公共数据集蒸馏训练轮数由1变为2






二、部署服务器问题归类：

1、 No such file or directory: '..\\OriginalHotspotDataset\\X_train_ICCAD.npy'

\ 改为 /

 2、Argument(s) not recognized: {'lr': 0.001}  ；lr改为learning_rate（Neural_Networks.py文件和FedMD.py文件）



三、终端命令细节
1、要将输出文件 ResultOutputData.md 保存到 ResultOutput 文件夹下，
并在文件名中添加当前时间属性，你可以使用以下命令。这里我们使用了 date 命令来生成当前时间，
并将其包含在文件名中：
mkdir -p ResultOutput  # 确保 ResultOutput 文件夹存在



2、使用 tmux 你在断开连接后继续运行任务,tmux 是一个终端复用器
使用步骤如下：
启动一个 tmux 会话：
tmux new -s lxzSession10NodesAsyn    tmux new -s lxzSession10NodesSyn


断开 tmux 会话（不终止会话），按下 Ctrl+B 然后按 D。
要重新连接到会话：
tmux attach -t lxzSession10NodesAsyn

tmux attach -t lxzSession10NodesSyn
tmux attach -t mysession
这几种方法都可以确保你的任务在断开连接后继续运行。选择适合你需求的方法即可。

3、 在 tmux 会话中运行你的命令：
#以下终端语句表示10个节点，50%异步率和100%同步率，每个节点（包含induatry节点）私有拥有原总数据量（iccad的18000）的5%（即900），
公共数据集拥有原总数据量（iccad的18000）的50%即9000，dataSetReductPara-2参数为2表示训练集一分为2, 
round-50表示训练50轮次,iccad、industry训练集取第一部分，iccad公共数据集取iccad总训练集第二部分：

如果你想使用 GPU 0 来运行你的代码，这样做,查看当前 GPU 的使用情况和状态:  nvidia-smi 使用空闲gpu：
CUDA_VISIBLE_DEVICES=6 python FConvConsistFedMD_Balanced1A100.py > ResultOutput/ResultOutputData_10Nodes100%Syn50%Asyn10%PrivateDate50%PublicData2DataSetReductPara50RoundGetTimeIs_$(date +"%Y-%m-%d_%H-%M-%S").md






四、训练集和测试集相关信息  和重要参数注释
训练集 iccad 热点1204    非热点 17096  总18000   
    industry 热点3629  非热点 80299   总84000   

测试集 iccad 热点2524  非热点 138848
    industry 热点942 非热点 20412

ConvConsistFedMD_balance_conf_annotation：
{
    "Annotation": "这个文件只做注释用，同时存储原始的 数据值",
    "N_parties": 10, 用途: 表示有多少个独立的实体在进行联邦学习。
    "N_samples_per_class": 1600,  用途: 控制每个参与方私有的用于训练的每个类别的样本量，从而影响模型的多样性和训练效果。
    "N_alignment": 30000, 用途: 用于在参与方之间对齐共享的数据集大小，用于对齐步骤或公共数据的训练。
    "private_classes": [0,1],
    "public_classes": [0, 1],
    "is_show": False,
    "N_rounds": 50, 用途: 控制整个联邦学习过程进行多少轮次的全局更新。一般25轮次就够用了
    "N_logits_matching_round": 2,  用途: 指定在每轮联邦学习中，进行logits匹配的次数，logits匹配是指对齐不同参与方模型输出的一种方法。
    "N_private_training_round": 10, 用途: 指定每轮联邦学习中，每个参与方对其私有数据进行模型训练的次数。
    "private_training_batchsize" : 64,
    "asynchronousRate": 0.5,
    "logits_matching_batchsize": 128, 用途: 控制在进行logits匹配时，每次处理的数据量。
    "EMNIST_dir": "./dataset/emnist-letters.mat",  用途: 指定用于联邦学习的数据集文件路径，目前没用这个参数。
    "model_saved_dir": "./pretrained_from_MNIST/", 用途: 指定用于存储预训练模型的目录路径。
    "result_save_dir": "./FEMNIST_balanced/", 用途: 指定联邦学习过程中生成的结果和模型的保存目录路径。
}

五、报错相关：
5.1、INTERNAL: libdevice not found at ./libdevice.10.bc  Aborted (core dumped)
解决：
export PATH=/usr/local/cuda-12.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.0


# 参考FedMD
FedMD: Heterogenous Federated Learning via Model Distillation. 
Preprint on https://arxiv.org/abs/1910.03581.



## Run scripts on Google Colab
1. open a google Colab

2. Clone the project folder from Github
```
! git clone github_link
```

3. Then access the folder just created. 
```
% cd project_folder/
```

4. Run the python script in Colab. For instance 
``` python
! python pretrain_CNN_on_public_dataset.py -conf conf/pretrain_MNIST_conf.json
```


其他：
import time

start_time = time.time() 
# 要统计执行时间的代码
end_time = time.time()
print(f"执行时间: {end_time - start_time:.6f} 秒")


源程序改动的部分（可删除）:
要改的部分（EMNIST和emnist改为ConvConsistFedMD）
"conf/EMNIST_balance_conf.json"
"./dataset/emnist-letters.mat",（这个可以先不改；因为没用到这个，直接删除）
"./FEMNIST_balanced/"
"..\OriginalHotspotDataset\_EMNIST_**.npy"（这里的得全改）


删除 FEMNIST_Balanced.py 文件【mnist的源文件】
注意MNIST的替换问题