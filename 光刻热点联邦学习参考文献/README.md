## 环境要求

所用的layout hotspot/non-hotspot数据应当只存在于CUHK的服务器上。
所用的`conda`环境配置见`environment.yml`文件。

### CUHK服务器相关

要用GPU需要运行申请命令(后缀是时长，以小时计)：
- `gogpu1_2`
- `gogpu1_8`
- `gogpu1_24`
使用完毕后请及时释放资源，如使用`exit`命令退出GPU环境。

激活配置好的`conda`环境：
```bash
conda activate jp_hs_env
```

## 运行训练程序

### 本地参数为FC2

将最后一层全连接层（`FC2`）作为本地参数进行训练：
```bash
python trainval_with_local_fc2.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --top-k-channels 26 \
    --benchmark_path /research/byu2/rchen/proj/wwy/jp/benchmarks \
    --model_path    <your model path>
```
`--n_iccad2012`和`--n_asml1`分别给定了`ICCAD2012`和`ASML1`两个数据集所分的client的数量；
`--sel_ratio`给定了没轮随机选择参与训练的clients占所有clients的比例；
`--top-k-channels`给定了选择的最优的channels的数量，即先将channels按照重要性排序，选择最优的`k`个，实例中`k=26`；
`--benchmark_path`给定benchmark数据（csv文件格式）的路径；
`--model_path`给定存储模型的路径，运行时将`<your model path>`改成自行定义的合法路径。

### 本地参数为FC1-2

将最后两层全连接层（`FC1-2`）作为本地参数进行训练：
```bash
python trainval_with_local_fc12.py \
    --n_iccad2012   2 \
    --n_asml1       2 \
    --sel_ratio     0.5 \
    --fc1-size  250 200 50 100 \
    --benchmark_path /research/byu2/rchen/proj/wwy/jp/benchmarks \
    --model_path    <your model path>
```
`--n_iccad2012`和`--n_asml1`分别给定了`ICCAD2012`和`ASML1`两个数据集所分的client的数量；
`--sel_ratio`给定了没轮随机选择参与训练的clients占所有clients的比例；
`--fc1-size`给定每个client的倒数第二层，即第一层全连接层（`FC1`）的大小，顺序是先`asml1`再`iccad2012`，如实例中`250 200 50 100`指2个`asml1`的client的`FC1`大小分别为250、200，而2个`iccad2012`的client的`FC1`大小分别为50、100；
`--benchmark_path`给定benchmark数据（csv文件格式）的路径；
`--model_path`给定存储模型的路径，运行时将`<your model path>`改成自行定义的合法路径。

### 纯本地训练

每个client仅在自己的数据上训练：
```bash
python -u trainval_no_server.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --benchmark_path /research/byu2/rchen/proj/wwy/jp/benchmarks \
    --model_path    <your model path>
```

### 纯全局训练

不分配client，使用所有的data进行训练：
```bash
python -u trainval_single_model.py \
    --model_path    <your model path>
```

### FedProx训练

```bash
python -u trainval_global_fedprox.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --model_path    <your model path>
```

## 画图程序

画图程序其实就是读取训练时输出的log，将打印出来的数值按序列画成曲线。
`plot/`路径下存放画图程序。主要是`async.py`画异步实验的曲线，`sync.py`画同步实验的曲线。

要将训练时打印出来的信息存成log文件，可以用`tee`，如
```bash
python -u trainval_global_fedprox.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --model_path    <your model path> \
    | tee log/xx.log
```

