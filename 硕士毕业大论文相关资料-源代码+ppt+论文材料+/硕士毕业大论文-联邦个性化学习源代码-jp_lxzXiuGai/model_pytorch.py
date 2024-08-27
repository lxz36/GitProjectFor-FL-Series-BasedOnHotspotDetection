import torch
import torch.nn as nn
import torch.nn.functional as F

#有修改这里储存着训练的模型

class Model(nn.Module):   #用于iccad
    def __init__(self, n_features=32, fc1_size=250):
        super(Model, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_features, 16, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*32, fc1_size, bias=True),
            nn.ReLU(True))
        self.final_fc = nn.Linear(fc1_size, 2, bias=True)

        self._initialize_weights()

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.pool1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.pool2(out)

        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.final_fc(out)

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
#有修改这里储存着训练的模型
class Model1(nn.Module):   #用于asml
    def __init__(self, n_features=32, fc1_size=250):
        super(Model1, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(n_features, 16, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True),
            nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*32, fc1_size, bias=True),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_size, fc1_size, bias=True),
            nn.ReLU(True))
        self.final_fc = nn.Linear(fc1_size, 2, bias=True)

        self._initialize_weights()

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.pool1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.pool2(out)

        out = out.flatten(start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.final_fc(out)

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



