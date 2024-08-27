# 示例参数
acc_ICCAD_list = [0.85, 0.88, 0.90]  # Accuracy 列表
tpr_ICCAD_list = [0.75, 0.78, 0.80]  # True Positive Rate 列表
pos_ICCAD_rate = 0.4  # 正样本比例

# 计算 FPR 列表
fpr_FAB_list = calculate_fpr(acc_ICCAD_list, tpr_ICCAD_list, pos_ICCAD_rate)

# 输出结果
for i, fpr in enumerate(fpr_FAB_list):
    print(f"Case {i+1} False Positive Rate (FPR): {fpr:.4f}")