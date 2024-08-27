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

#起源；所有的注释都在这里，这里是直接做的对热区图片的复原；尽量精简程序；已经把神经网络设置为6layer；
# 查看形状  shape  size
#目前的实验证明：确实参数恢复不了数据
#输入的图片一定是jpg； 每次只恢复一张照片出效果是最好的；最好输入也只有一张照片


class LeNet(nn.Module):  #定义神经网络的架构
    def __init__(self, channel=3, hideen=32768, num_classes=2):

        hideen = 32768  #全连接层第一层的神经元数量
        hideenEndLayer = 250  #全连接层最后一层的神经元数量  初始值250
        num_classes = 2
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=3 // 2, stride=1),    #input，filter是设定的输入和卷积核，参数strides，padding分别决定了卷积操作中滑动步长和图像边沿填充的方式。
            act(),
            nn.Conv2d(16, 16, kernel_size=3, padding=3 // 2, stride=1),  #  计算公式output_size长宽 =( input_size长宽 + padding * 2 - kernel_size ) / strides + 1
            act(),
            nn.Conv2d(16, 32, kernel_size=3, padding=3 // 2, stride=1),
            act(),
            nn.Conv2d(32, 32, kernel_size=3, padding=3 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(

            nn.Linear(hideen, hideenEndLayer),
            nn.Linear(hideenEndLayer, num_classes)
        )

    def forward(self, x):
        # print("x.ndim:",x.ndim)
        # print("x.shape:",x.shape)
        out = self.body(x)
        # print("out1.shape:", out.shape)
        out = out.view(out.size(0), -1)
        # print("out2.shape:", out.shape)
        out = self.fc(out)

        # print("out3.shape:", out.shape)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
#

class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst
#

# #加噪声的环节
# def RecursionAddNoise(n):    #通过递归的方式给不规则list列表中每个元素加噪声
#     for i in range(len(n)):
#         if type(n[i]) is list:
#             RecursionAddNoise(n[i])
#         else:
#
#             for j in range(len(n[i])):
#                 noiseValue = np.random.laplace(0, 0.0001, 1) #最小刻度的噪声
#                 n[i][j]=n[i][j]+noiseValue#2就是安全多方计算的随机数字


# 优化加噪声的环节1，给每个tensor阵加固定噪声
# def RecursionAddNoise(n):  # 通过递归的方式给不规则list列表中每个元素加噪声
#     if type(n[0]) is list:
#         for i in range(len(n)):
#             RecursionAddNoise(n[i])
#     else:
#         for j in range(len(n)):
#
#             noiseValue = np.random.laplace(0, 0.0001, 1)  # 最小刻度的噪声
#             n[j] = n[j] + noiseValue  # 2就是安全多方计算的随机数字
#
#
#
#             print("原始值：",n[j] )
#             print("原始jlist值：", n)
#             print("加入的噪声：",noiseValue )
#             print("噪声jlist值：", n)
#             if j==2:
#                 break

# 优化加噪声的环节2，给每个元素加固定噪声
def RecursionAddNoise(n):  # 通过递归的方式给不规则list列表中每个元素加噪声
    if type(n[0]) is list:
        for i in range(len(n)):
            RecursionAddNoise(n[i])
    else:
        print("查看Tensor维度:",n[0].shape)  # 使用shape查看Tensor维度
        print(n[0].numel())      #numel()函数，查看一个张量有多少元素

        # noiseValue=np.random.laplace(0,0.1,[16,3,3,3])
        noiseValue=np.random.laplace(0,1,[2,2,2])  #list(n[0].shape)  将torch的维度转化为list的维度
        noiseValue1=np.random.laplace(0,1,[2,2,2])  #list(n[0].shape)  将torch的维度转化为list的维度
        lina=torch.Tensor(noiseValue)
        print("lina:",lina)
        linb=torch.Tensor(noiseValue1)
        print("linb:",linb)

        linc=lina+linb
        print("linc:",linc)

        print("查看noiseValue的Tensor维度:", torch.Tensor(noiseValue).shape)  # 使用shape查看Tensor维度
        # for j in range(len(n)):
        #
        #     noiseValue = np.random.laplace(0, 0.0001, 1)  # 最小刻度的噪声
        #     n[j] = n[j] + noiseValue  # 2就是安全多方计算的随机数字
        #
        #
        #
        #     print("原始值：",n[j] )
        #     print("原始jlist值：", n)
        #     print("加入的噪声：",noiseValue )
        #     print("噪声jlist值：", n)
        #     if j==2:
        #         break





def main():


    dataset = 'lfw'
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s' % dataset).replace('\\', '/')


    #这里有一些全局参数设置
    lr = 1.0
    num_dummy = 1    #每次实验一口气生成照片的数量？
    Iteration = 100 #initial Value=300  每隔多少次打印一下结果   总的训练次数
    num_exp = 3  #initial Value=1000  这里代表最终总共生成复原多少张图片  每次只恢复一张照片出效果是最好的；最好输入也只有一张照片

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print("device is: ",device)

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)



    ''' load data '''

    if dataset == 'lfw':    #我们把这个作为热区检测得一些全局参数设置
        shape_img = (32, 32)
        num_classes = 2
        channel = 3
        hidden = 32768
        lfw_path = os.path.join(root_path, './data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')




    ''' train DLG and iDLG '''
    for idx_net in range(num_exp):
        # net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes)  #这里修改传入得参数设置 输入维度和输出维度
        net = LeNet(channel=3, hideen=hidden, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        # for method in ['DLG', 'iDLG']:   #只做idlg一种情况
        for method in [ 'iDLG']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            print("dy_dx.shape:", len(dy_dx), len(dy_dx[10]),dy_dx[10][0].shape)
            # for _ in dy_dx:
            #     print(len(_))
            #     print(_[0].shape)
            # for i in range(len(dy_dx)):
            #     if i!=10:
            #         original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            #     else:

            # original_dy_dx = list((dy_dx[i].detach().clone() if i != 10 else (dy_dx[i]*0.5).detach().clone() for i in range(len(dy_dx))))  #这里最后一层的参数or梯度全部置为0
            original_dy_dx = list((dy_dx[i].detach().clone() if i != 10 else (dy_dx[i]).detach().clone() for i in range(len(dy_dx))))  #这里最后一层的参数or梯度全部置为原数据
            # original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            print("original_dy_dx.shape:", len(original_dy_dx))

            #调试如下，防御措施
            print('加入防御措施前的original_dy_dx[0]:',original_dy_dx[0][0])
            RecursionAddNoise(original_dy_dx)
            print('加入防御措施后的original_dy_dx[0]:',original_dy_dx[0][0])
            # 调试如上，防御措施


            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

            if method == 'DLG':
                optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
            elif method == 'iDLG':
                optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                # predict the ground-truth label
                label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []

            print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    if method == 'DLG':
                        dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                        # dummy_loss = criterion(pred, gt_label)
                    elif method == 'iDLG':
                        dummy_loss = criterion(pred, label_pred)

                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    grad_diff = 0

                    for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                        grad_diff += ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff

                optimizer.step(closure)
                current_loss = closure().item()
                train_iters.append(iters)
                losses.append(current_loss)
                mses.append(torch.mean((dummy_data-gt_data)**2).item())


                if iters % int(Iteration / 30) == 0:  #每隔一段时间生成一个小图片
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'DLG':
                            plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG':
                            plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

                    # if current_loss < 0.000001: # converge  #当复原图片和真实图片的差值很小时；停止训练
                    #     break

            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses



        print('imidx_list:', imidx_list)

        # print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])#dlg的效果
        print( 'mse_iDLG:', mse_iDLG[-1])
        # print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)#dlg的效果
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')

if __name__ == '__main__':
    main()

#
