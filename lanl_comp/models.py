import torch.nn as nn
# import torchvision
# from torchvision import models as torch_models


class BaselineNetOneChannel(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 1, 2, stride=1),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+1),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+2),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+4),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+8),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+16),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+32),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+64),
            nn.ReLU(),

            nn.Conv1d(1, 1, 2, stride=2, dilation=1+32)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetThreeChannel(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 3, 2, stride=1),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+1),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+2),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+4),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+8),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+16),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+32),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+64),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+128),
            nn.ReLU(),

            nn.Conv1d(3, 3, 2, stride=2, dilation=1+256),
            nn.ReLU(),

            nn.Conv1d(3, 1, 2, stride=2),
            nn.ReLU(),

            nn.Linear(61, 1)

            # nn.Conv1d(3, 1, 2, stride=2, dilation=1+32)
        )

    def forward(self, inpt):
        return self.body(inpt)


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
