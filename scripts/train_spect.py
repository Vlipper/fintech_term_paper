import os
# os.environ['MKL_NUM_THREADS'] = '1'
from datetime import datetime

import pandas as pd
import numpy as np

import torch.nn as nn
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
# data_path = '/mntlong/lanl_comp/data/'
file_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'data'))
train_info_path = os.path.join(data_path, 'train_info.csv')
train_data_path = os.path.join(data_path, 'train_compressed.npz')

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

# training params
model_name = 'spectr_net_v1'
batch_size = 100  # 1300
num_epochs = 5

hz_cutoff = 600000  # {0, ..., 600000, ...}
window_size = 150000
overlap_size = int(window_size * 0.5)
nperseg = 2048

num_bins = 17

# get modified resnet model
# model = models.BaselineNetSpect()
model = tv_models.resnet34(pretrained=True)
model = models.get_resnet(model, out_size=num_bins-1)
loss_fn = nn.CrossEntropyLoss()

# logs_path = '/mntlong/lanl_comp/logs/'
logs_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'logs'))
current_datetime = datetime.today().strftime('%b-%d_%H-%M-%S')
log_writer_path = os.path.join(logs_path, 'runs', current_datetime + '_' + model_name)

train_dataset = data.SpectrogramDataset(train_signal, train_quaketime, num_bins=num_bins,
                                        idxs_wave_end=train_info['indx_end'].values,
                                        hz_cutoff=hz_cutoff, window_size=window_size,
                                        overlap_size=overlap_size, nperseg=nperseg)
val_dataset = data.SpectrogramDataset(val_signal, val_quaketime, num_bins=num_bins,
                                      idxs_wave_end=train_info['indx_end'].values,
                                      hz_cutoff=hz_cutoff, window_size=window_size,
                                      overlap_size=overlap_size, nperseg=nperseg)

print('spectrogram size:', train_dataset[0][0].size())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=5,
                          pin_memory=True)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=5,
                        pin_memory=True)

opt = optim.Adam(model.parameters(), lr=3e-4)
lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, threshold=0.001)
log_writer = SummaryWriter(log_writer_path)

utils.train_spec_model(model=model, optimizer=opt, lr_scheduler=lr_sched,
                       train_loader=train_loader, val_loader=val_loader,
                       num_epochs=num_epochs, model_name=model_name,
                       logs_path=logs_path, log_writer=log_writer,
                       loss_fn=loss_fn, num_bins=num_bins)
