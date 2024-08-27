

def calculate_fpr(acc_list, tpr_list, pos_rate):
    # pos_rate 是正样本比例 (pos_rate = (TP + FN) / Total)
    neg_rate = 1 - pos_rate

    # 创建一个空列表存储结果
    fpr_list = []

    # 迭代遍历 acc 和 tpr 列表
    for acc, tpr in zip(acc_list, tpr_list):
        # 根据 ACC 和 TPR 计算 TNR
        tnr = (acc - (tpr * pos_rate)) / neg_rate

        # 根据 TNR 计算 FPR
        fpr = 1 - tnr

        # 将 FPR 添加到结果列表中
        fpr_list.append(fpr)

    return fpr_list

# 示例参数
acc_ICCAD_list = [0.9156, 0.88, 0.90       ]  # Accuracy 列表  前面是同步，后面是异步
tpr_ICCAD_list = [0.9869, 0.78, 0.80]  # True Positive Rate 列表
pos_ICCAD_rate = 2524.0/ (138848 +2524.0) # 正样本比例

# 计算 FPR 列表
fpr_ICCAD_list = calculate_fpr(acc_ICCAD_list, tpr_ICCAD_list, pos_ICCAD_rate)

# 输出结果
for i, fpr in enumerate(fpr_ICCAD_list):
    print(f"Case {i+1} False Positive Rate (FPR): {fpr:.4f}")

print(fpr_ICCAD_list)


# 示例参数
acc_FAB_list = [0.9156,
0.9097,
0.8456,
0.9383,
0.9567,
0.8954,
0.8281,
0.9198,
0.9564,
0.8231,
0.5584,
0.7252,
0.8698,
0.8889,
0.6091,
0.7296,
0.8277,
0.8881,

0.5584

]  # Accuracy 列表
tpr_FAB_list = [0.9869,
0.9294,
0.9643,
0.9865,
0.9901,
0.9809,
0.7892,
0.9885,
0.9896,
0.8591,
0.3602,
0.6945,
0.8448,
0.9396,
0.6916,
0.6524,
0.8556,
0.9071,

0.5802
]  # True Positive Rate 列表
pos_FAB_rate = 942.0/ (942+20412) # 正样本比例

# 计算 FPR 列表
fpr_FAB_list = calculate_fpr(acc_FAB_list, tpr_FAB_list, pos_FAB_rate)

# 输出结果
for i, fpr in enumerate(fpr_FAB_list):
    print(f"Case {i+1} False Positive Rate (FPR): {fpr:.4f}")


# 按列输出 FPR 列表
for fpr in fpr_FAB_list:
    print(f"{fpr:.4f}")

