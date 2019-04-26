import numpy as np
import pandas as pd
from sklearn import linear_model

linear_model.LinearRegression()

pd.read_csv()

train_signal = np.load('../data/train_compressed.npz')['signal']
train_signal = np.array(train_signal)

print(train_signal.shape)

from tensorboardX import SummaryWriter

# инициализация
writer_tr = SummaryWriter()
writer_tst = SummaryWriter(stats_folder + '/test')

writer_tr.add_scalars()

# внутри цикла
writer_tr.add_scalar('Loss/Total loss', np.array(10), 1)
writer_tst.add_scalar('Loss/Total loss',val_loss , epoch)


#%% tests
import importlib
importlib.reload(data)
import data

tst = data.SpectrogramDataset(train_signal, train_quaketime,
                              hz_cutoff=0, exmpl_size=10000)  # 10**6

tst_out = tst[0]
tst_out[0].size()

tst_out[0].view(1, 1, 65, 89).size()


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


# tst_resnet = tv_models.resnet18(False)
tst_model = BaselineNetSpect()
tst_model_out = tst_model.forward(tst_out[0].view(1, 1, 65, 89))
tst_model_out


tst_line = nn.Linear(64, 1)

tst_model_out.size()
tst_line(tst_model_out).size()

#%% end_tests



# Dataset
# class SignalTestDataset(Dataset):
#     def __init__(self, files_path, file_names_list):
#         super().__init__()
#
#         self.files_path = files_path
#         self.file_names_list = file_names_list
#
#     def __len__(self):
#         return len(self.file_names_list)
#
#     def __getitem__(self, index):
#         file = np.loadtxt(self.files_path + self.file_names_list[index],
#                           dtype=np.float32,
#                           skiprows=1 + 140000)  # skiprows temp decision
#         return torch.from_numpy(file).view(1, -1)

window_size = 100
wave_size = 1000
overlap_size = window_size // 2
# n = 3  # group size
# m = 1  # overlap size

# n = 1
# tst = [(i * window_size + n, (i + 1) * window_size + n)
#        for i in range(wave_size // window_size + 1)]

tst = [(i, i + window_size)
       for i in range(0, wave_size, window_size - overlap_size)
       if i + window_size <= wave_size]
tst


from itertools import islice
import numpy as np

tst = islice(np.arange(10), 2)
