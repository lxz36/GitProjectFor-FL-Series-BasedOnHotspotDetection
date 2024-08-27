

#时间规划，思路顺序
1、先把卷积层加入到损失函数
2、改造为可以调整节点数量、数据集大小、同步异步的统一程序

2.2调整数据集大小：注意调整ConvConsistFedMD_balance_conf.json文件的N_samples_per_class参数（原始值为8000）和
"private_training_batchsize"参数（原始值为64）和"N_alignment"（原始值为30000）,

#目前整个工程是改造成为光刻热区的联邦蒸馏检测模型
FConvConsistFedMD_Balanced1 {(6层CNN神经网络)}是**主程序**
pretrain_CNN_on_public_dataset是预先验证的程序，产生预训练模型



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

5、运行蒸馏程序
python FConvConsistFedMD_Balanced1A100.py




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