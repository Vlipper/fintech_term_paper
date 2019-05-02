import pandas as pd
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import BatchSampler # Sampler,
from torchvision import models as tv_models
from tensorboardX import SummaryWriter

# import local scripts
import data
import models
import utils


# load train data into ram
source_path = '/mntlong/lanl_comp/data/'
train_info_path = source_path + 'train_info.csv'
train_data_path = source_path + 'train_compressed.npz'

train_info = pd.read_csv(train_info_path, index_col='Unnamed: 0')
train_info['exp_len'] = train_info['indx_end'] - train_info['indx_start']

train_signal = np.load(train_data_path)['signal']
train_quaketime = np.load(train_data_path)['quake_time']

# В валидацию берем 2 последних волны (части эксперимента)
val_start_idx = train_info.iloc[-2, :]['indx_start']

val_signal = train_signal[val_start_idx:]
val_quaketime = train_quaketime[val_start_idx:]

train_signal = train_signal[:val_start_idx]
train_quaketime = train_quaketime[:val_start_idx]

# get modified resnet model
model = tv_models.resnet34(pretrained=True)
model = models.get_resnet(model)

# training process
logs_path = '/mntlong/lanl_comp/logs/'
batch_size = 1300
num_epochs = 10
model_name = 'spectr_net_v1'
window_size = 10000
overlap_size = window_size // 2

train_loader = DataLoader(
    dataset=data.SpectrogramDataset(train_signal, train_quaketime,
                                    hz_cutoff=700000, window_size=window_size,
                                    overlap_size=overlap_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=5,
    pin_memory=True)
val_loader = DataLoader(
    dataset=data.SpectrogramDataset(val_signal, val_quaketime,
                                    hz_cutoff=0, window_size=window_size,
                                    overlap_size=overlap_size),
    batch_size=batch_size,
    shuffle=False,
    num_workers=5,
    pin_memory=True)

# model = models.BaselineNetSpect()
opt = optim.Adam(model.parameters(), lr=1e-2)
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, threshold=0.001)
log_writer = SummaryWriter()

utils.train_model(model=model, optimizer=opt, lr_scheduler=lr_sched,
                  train_loader=train_loader, val_loader=val_loader,
                  num_epochs=num_epochs, model_name=model_name,
                  logs_path=logs_path, log_writer=log_writer)
