import numpy as np
import os

# 存储asml1测试集
X_test_ConvConsistFedMD_asml1 = np.load(
    file="..\OriginalHotspotDataset\X_test_ConvConsistFedMD_asml1.npy")
y_test_ConvConsistFedMD_asml1 = np.load(
    file="..\OriginalHotspotDataset\y_test_ConvConsistFedMD_asml1.npy")


# 加载原始数据集
X_train_ICCAD = X_test_ConvConsistFedMD_asml1
y_train_ICCAD = y_test_ConvConsistFedMD_asml1

# 确定缩小后的数据集大小
n_samples = len(X_train_ICCAD) // 10

# 选择数据集的一部分
X_train_ICCAD_reduced = X_train_ICCAD[:n_samples]
y_train_ICCAD_reduced = y_train_ICCAD[:n_samples]

# 保存缩小后的数据集到指定文件夹
output_dir = "..\OriginalHotspotDatasetSmall"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "X_test_ConvConsistFedMD_asml1.npy"), X_train_ICCAD_reduced)
np.save(os.path.join(output_dir, "y_test_ConvConsistFedMD_asml1.npy"), y_train_ICCAD_reduced)

print("数据集已成功缩小并保存到指定文件夹。")
