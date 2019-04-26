import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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

train_signal = torch.from_numpy(train_signal)
train_quaketime = torch.from_numpy(train_quaketime)

# В валидацию берем 2 последних волны (части эксперимента)
val_start_idx = train_info.iloc[-2, :]['indx_start']

val_signal = train_signal[val_start_idx:]
val_quaketime = train_quaketime[val_start_idx:]

train_signal = train_signal[:val_start_idx]
train_quaketime = train_quaketime[:val_start_idx]


# training process
logs_path = '/mntlong/lanl_comp/logs/'
batch_size = 10000
model_name = 'simple_net_v2'
window_size = 10000

train_loader = DataLoader(
    dataset=data.SignalDataset(train_signal, train_quaketime, window_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True)
val_loader = DataLoader(
    dataset=data.SignalDataset(val_signal, val_quaketime, window_size),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

simple_net = models.BaselineNetOneChannel
opt = optim.Adam(simple_net.parameters(), lr=1e-2)
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, threshold=0.001)

utils.train_model(model=simple_net, optimizer=opt, lr_scheduler=lr_sched,
                  train_loader=train_loader, val_loader=val_loader,
                  num_epochs=500, model_name=model_name, logs_path=logs_path)
