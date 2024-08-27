#描述的是sensitivety与噪声方差的关系
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image

def noisyCount(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value
def laplace_mech(data, sensitivety, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety, epsilon)
    return data

Laplace_noise = np.zeros([1000], dtype=float)
# 噪声初始化
sensitivety = 0.073   #sensitivety = 0.00725 对应噪声e-4   0.0715  对应噪声e-3  0.073  对应噪声e-2
epsilon = 1
Laplace_noise = laplace_mech(Laplace_noise, sensitivety, epsilon)

arr_var = np.var(Laplace_noise)
print(arr_var)