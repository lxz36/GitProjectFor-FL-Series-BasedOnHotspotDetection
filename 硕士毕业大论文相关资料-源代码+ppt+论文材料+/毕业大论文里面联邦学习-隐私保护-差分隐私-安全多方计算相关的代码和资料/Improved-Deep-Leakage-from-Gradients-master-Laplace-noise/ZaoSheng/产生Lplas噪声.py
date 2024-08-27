import numpy as np


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


if __name__ == '__main__':
    Laplace_noise = np.zeros([1000],dtype=float)#噪声初始化
    sensitivety = 1
    epsilon = 1
    Laplace_noise = laplace_mech(Laplace_noise, sensitivety, epsilon)
    print('data:',Laplace_noise)

