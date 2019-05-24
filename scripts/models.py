from collections import OrderedDict
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BaselineNetRawSignalV1(nn.Module):
    def __init__(self, out_size):
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

            nn.Conv1d(32, 32, 2, stride=2, dilation=1+16),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 2, stride=2, dilation=1+64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 2, stride=2, dilation=1+128),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 2, stride=2, dilation=1+256),
            nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True,

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(in_features=128, out_features=out_size, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetRawSignalV2(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 16, 2, stride=1),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+1),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 16, 2, stride=2, dilation=1+2),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),

            nn.Conv1d(16, 32, 2, stride=2, dilation=1+4),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),

            nn.Conv1d(32, 64, 2, stride=2, dilation=1+8),
            nn.Dropout(0.2),
            nn.MaxPool1d(3),

            # nn.Conv1d(64, 128, 2, stride=2, dilation=1+16),
            # nn.Dropout(0.5),
            # nn.MaxPool1d(3),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(in_features=64, out_features=out_size, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetRawSignalV3(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 10, stride=1, bias=False),
            nn.ReLU(),

            nn.MaxPool1d(100),

            nn.Conv1d(32, 64, 10, stride=1, bias=False),
            nn.ReLU(),

            nn.Conv1d(64, 128, 10, stride=1, bias=False),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),

            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=out_size, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetRawSignalV4(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv1d(1, 64, 4000, stride=10),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, 3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, 3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),

            nn.Linear(in_features=512, out_features=out_size, bias=True)
        )

    def forward(self, inpt):
        return self.body(inpt)


class BaselineNetRawSignalV5(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.input = nn.Sequential(nn.Conv1d(1, 64, 4000, stride=10, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(4))

        self.layer1_1 = nn.Sequential(nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU())
        self.layer1_2 = nn.Sequential(nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm1d(64),
                                      nn.ReLU())

        self.layer2_1 = nn.Sequential(nn.Conv1d(64, 256, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU())
        self.layer2_2 = nn.Sequential(nn.Conv1d(256, 256, 3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm1d(256),
                                      nn.ReLU())

        self.tofc = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Flatten())

        self.fc = nn.Linear(in_features=256, out_features=out_size, bias=True)

        self.relu = nn.ReLU()

        self.downsample2 = nn.Sequential(nn.Conv1d(64, 256, 1, stride=1, bias=False),
                                         nn.BatchNorm1d(256))

    def forward(self, x):
        out = self.input(x)

        skip = out
        out = self.layer1_1(out)
        out = self.layer1_2(out)
        out += skip
        # out = self.relu(out)

        skip = out
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        out += self.downsample2(skip)

        out = self.tofc(out)
        # out = self.fc(out)

        return out


class BaselineNetRawSignalCnnRnnV1(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 4000, stride=10, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(32, 64, 3, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Conv1d(64, 64, 3, stride=1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 128, 3, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(128, 128, 3, stride=1),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(128, 256, 3, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Conv1d(256, 256, 3, stride=1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.MaxPool1d(4),

            # nn.Conv1d(256, 512, 3, stride=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Conv1d(512, 512, 3, stride=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            Flatten()
        )

        # self.rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=2,
        #                   nonlinearity='relu', batch_first=True)
        self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=10,
                           batch_first=True)

        self.fc = nn.Linear(in_features=512, out_features=out_size, bias=True)

    def forward(self, inpt):
        out = self.cnn(inpt)

        out = out.view(-1, 1, 256)
        out = self.rnn(out)[0]
        out = out.squeeze(1)

        out = self.fc(out)
        return out


class RawSignalCnnRnnV2(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 4000, stride=10, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(32, 64, 3, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 128, 3, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(128, 256, 3, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, 3, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            Flatten()
        )

        # self.rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=2,
        #                   nonlinearity='relu', batch_first=True)
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=10,
                           batch_first=True)

        self.fc = nn.Linear(in_features=512, out_features=out_size, bias=True)

    def forward(self, inpt):
        out = self.cnn(inpt)

        out = out.view(-1, 1, 256)
        out = self.rnn(out)[0]
        out = out.squeeze(1)

        out = self.fc(out)
        return out


# class CnnRnnV2(nn.Module):
#     def __init__(self, out_size):
#         super().__init__()
#
#         self.cnn = BaselineNetRawSignalV5(1)
#
#         # self.rnn = nn.RNN(input_size=256, hidden_size=512, num_layers=2,
#         #                   nonlinearity='relu', batch_first=True)
#         self.rnn = nn.LSTM(input_size=256, hidden_size=512, num_layers=10,
#                            batch_first=True)
#
#         self.fc = nn.Linear(in_features=512, out_features=out_size, bias=True)
#
#     def forward(self, inpt):
#         out = self.cnn(inpt)
#
#         out = out.view(-1, 1, 256)
#         out = self.rnn(out)[0]
#         # out = out.view(-1, 512)
#         out = out.squeeze(1)
#
#         out = self.fc(out)
#         return out


# Spectrogram models
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


def get_resnet(torchvision_resnet, out_size):
    modules = [*torchvision_resnet.named_children()]

    # layers modification
    modules.insert(-1, ('flatten', Flatten()))
    modules = OrderedDict(modules)
    modules['conv1'] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                 padding=(3, 3), bias=False)
    del modules['maxpool']
    modules['fc'] = nn.Linear(in_features=modules['fc'].in_features,
                              out_features=out_size,
                              bias=True)

    resnet_mod = nn.Sequential(modules)
    return resnet_mod
