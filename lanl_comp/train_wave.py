from datetime import datetime

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

train_signal = torch.from_numpy(train_signal)
train_quaketime = torch.from_numpy(train_quaketime)

# В валидацию берем 2 последних волны (части эксперимента)
val_start_idx = train_info.iloc[-2, :]['indx_start']

val_signal = train_signal[val_start_idx:]
val_quaketime = train_quaketime[val_start_idx:]

train_signal = train_signal[:val_start_idx]
train_quaketime = train_quaketime[:val_start_idx]

# training params
model_name = 'wave_net_v1_new2'
batch_size = 250
num_epochs = 100

window_size = 150000
overlap_size = int(window_size * 0.5)

model = models.BaselineNetRawSignalV2()
loss_fn = nn.SmoothL1Loss()  # L1Loss() SmoothL1Loss() MSELoss()

logs_path = '/mntlong/lanl_comp/logs/'
current_datetime = datetime.today().strftime('%b-%d_%H-%M-%S')
log_writer_path = logs_path + 'runs/' + current_datetime + '_' + model_name

train_dataset = data.SignalDataset(train_signal, train_quaketime,
                                   window_size=window_size,
                                   overlap_size=overlap_size)
val_dataset = data.SignalDataset(val_signal, val_quaketime,
                                 window_size=window_size,
                                 overlap_size=overlap_size)

print('wave size:', train_dataset[0][0].size())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=6,
                        pin_memory=True)


opt = optim.Adam(model.parameters(), lr=1e-3)
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, threshold=0.001)
log_writer = SummaryWriter(log_writer_path)

utils.train_model(model=model, optimizer=opt, lr_scheduler=lr_sched,
                  train_loader=train_loader, val_loader=val_loader,
                  num_epochs=num_epochs, model_name=model_name,
                  logs_path=logs_path, log_writer=log_writer,
                  loss_fn=loss_fn)
