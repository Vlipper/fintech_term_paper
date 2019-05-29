import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='spectr_net_default', type=str)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=150, type=int)
    parser.add_argument('--find_lr', default=False, action='store_true')
    return parser.parse_args()


def main(args):
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

    # spectrogram params
    nperseg = 2048
    hz_cutoff = 600000  # {0, ..., 600000, ...}
    window_size = 150000
    train_overlap_size = int(window_size * 0.5)
    val_overlap_size = int(window_size * 0.8)

    num_bins = 17  # 17
    wd = 1e-3  # 5e-4

    # get modified resnet model
    model = tv_models.resnet34(pretrained=True)
    model = models.get_resnet(model, out_size=num_bins-1)
    loss_fn = nn.CrossEntropyLoss()

    # logs_path = '/mntlong/lanl_comp/logs/'
    logs_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'logs'))
    current_datetime = datetime.today().strftime('%b-%d_%H-%M-%S')
    log_writer_path = os.path.join(logs_path, 'runs',
                                   current_datetime + '_' + args.model_name)

    train_dataset = data.SpectrogramDataset(train_signal, train_quaketime,
                                            num_bins=num_bins,
                                            idxs_wave_end=train_info['indx_end'].values,
                                            stdscale=True,
                                            hz_cutoff=hz_cutoff,
                                            window_size=window_size,
                                            overlap_size=train_overlap_size,
                                            nperseg=nperseg)
    val_dataset = data.SpectrogramDataset(val_signal, val_quaketime,
                                          num_bins=num_bins,
                                          idxs_wave_end=train_info['indx_end'].values,
                                          stdscale=True,
                                          hz_cutoff=hz_cutoff,
                                          window_size=window_size,
                                          overlap_size=val_overlap_size,
                                          nperseg=nperseg)
    print('spectrogram size:', train_dataset[0][0].size())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=5,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=5,
                            pin_memory=True)

    if args.find_lr:
        from lr_finder import LRFinder
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=wd)
        lr_find = LRFinder(model, optimizer, loss_fn, device='cuda')
        lr_find.range_test(train_loader, end_lr=1, num_iter=50, step_mode='exp')
        best_lr = lr_find.get_best_lr()
        lr_find.plot()
        lr_find.reset()
        print('best lr found: {:.2e}'.format(best_lr))
    else:
        best_lr = 3e-4

    optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=wd)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    factor=0.5,
                                                    patience=3,
                                                    threshold=0.01)
    log_writer = SummaryWriter(log_writer_path)

    utils.train_clf_model(model=model, optimizer=optimizer, lr_scheduler=lr_sched,
                          train_loader=train_loader, val_loader=val_loader,
                          num_epochs=args.num_epochs, model_name=args.model_name,
                          logs_path=logs_path, log_writer=log_writer,
                          loss_fn=loss_fn, num_bins=num_bins)


if __name__ == '__main__':
    args = parse_args()
    main(args)
