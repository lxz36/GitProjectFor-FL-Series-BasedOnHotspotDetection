from torchvision import datasets, models, transforms
from torch import nn


def resnet50_bcd():
    model = models.resnet50(pretrained=False)

    model.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )

    model.layer2[0].downsample = nn.Sequential(
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer3[0].downsample = nn.Sequential(
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer4[0].downsample = nn.Sequential(
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
        nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

    return model
