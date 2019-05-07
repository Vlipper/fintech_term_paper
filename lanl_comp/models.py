from collections import OrderedDict
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BaselineNetRawSignalV1(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 16, 2, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 2, stride=2, dilation=1+4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 32, 2, stride=2, dilation=1+8),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 2, stride=2, dilation=1+16),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 2, stride=2, dilation=1+32),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 2, stride=2, dilation=1+64),
            nn.BatchNorm1d(128),
            # nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetRawSignalV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 16, 2, stride=1),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+1),
            nn.Dropout(0.5),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+2),
            nn.Dropout(0.5),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 32, 2, stride=2, dilation=1+4),
            nn.Dropout(0.5),
            nn.MaxPool1d(3),

            nn.Conv1d(32, 64, 2, stride=2, dilation=1+8),
            nn.Dropout(0.5),
            nn.MaxPool1d(3),

            # nn.Conv1d(64, 128, 2, stride=2, dilation=1+16),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(3),
            #
            # nn.Conv1d(64, 64, 2, stride=2, dilation=1+32),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(3),

            # nn.Conv1d(64, 128, 2, stride=2, dilation=1+64),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(3),
            #
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(in_features=64, out_features=1, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)
# import torch
# model = BaselineNetRawSignalV2()
# inpt = torch.rand(1, 1, 25000)
# model(inpt).size()


class BaselineNetSpect(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.body(x)

        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def get_resnet(torchvision_resnet):
    modules = [*torchvision_resnet.named_children()]

    # layers modification
    modules.insert(-1, ('flatten', Flatten()))
    modules = OrderedDict(modules)
    modules['conv1'] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                 padding=(3, 3), bias=False)
    del modules['maxpool']
    modules['fc'] = nn.Linear(in_features=modules['fc'].in_features,
                              out_features=1,
                              bias=True)

    resnet_mod = nn.Sequential(modules)
    return resnet_mod
