import os
from tqdm import tqdm
import numpy as np
from scipy import signal

import torch
import torch.nn.functional as F
cuda = torch.device('cuda')
cpu = torch.device('cpu')


def spectrogram(sig_in, nperseg):
    """
    Params:
    sig_in -- input signal

    Return:
    f, t, np.log(Sxx + eps)
    """
    nperseg = nperseg  # default 256 -- размер окна
    noverlap = nperseg // 8
    fs = 4 * 1e+6  # raw signal sample rate is 4MHz
    window = 'hann'  # {triang, hamming}
    detrend = 'linear'  # {'linear', 'constant', False}
    scaling = 'density'  # {'density', 'spectrum'}
    eps = 1e-11
    f, t, Sxx = signal.spectrogram(sig_in, fs=fs, window=window,
                                   nperseg=nperseg, noverlap=noverlap,
                                   detrend=detrend, scaling=scaling)
    return f, t, np.log(Sxx + eps)


def other_wave_check(right_boarder, first_wave_windows):
    for next_wave_start, next_wave_end in first_wave_windows:
        if next_wave_start < right_boarder < next_wave_end:
            return False
    return True


def left_padding(row, needed_len):
    pad_size = needed_len - len(row)
    pad = np.zeros(pad_size)
    row_padded = np.concatenate((pad, row))

    return row_padded


def train_spec_model(model, optimizer, lr_scheduler, train_loader, val_loader,
                     num_epochs, model_name, logs_path, log_writer, loss_fn,
                     num_bins):
    model = model.to(cuda)
    torch_bins = torch.linspace(0, 16.11, num_bins, dtype=torch.float32)
    n_iter_train = 1
    for epoch in range(num_epochs):
        # training process
        model.train()
        for x, target, target_bin in tqdm(train_loader,
                                          desc='training, epoch={}'.format(epoch + 1),
                                          position=0):
            x = x.to(device=cuda, dtype=torch.float32, non_blocking=True)
            target = target.to(device=cuda, non_blocking=True)
            target_bin = target_bin.to(device=cuda, non_blocking=True)

            optimizer.zero_grad()
            out = model.forward(x)
            loss = loss_fn(out, target_bin)
            loss.backward()
            optimizer.step()

            # calc metrics
            max_out = torch.argmax(out, 1)
            bin_centroids = (torch_bins[max_out] + torch_bins[max_out + 1]) / 2
            bin_centroids = bin_centroids.to(device=cuda, non_blocking=True)
            metrics = F.l1_loss(bin_centroids, target)

            # logging
            log_writer.add_scalars('loss',
                                   {'train_batch': loss.item()},
                                   n_iter_train)
            log_writer.add_scalars('metrics_MAE',
                                   {'train_batch': metrics.item()},
                                   n_iter_train)

            # grads norms
            # grads_norms = {}
            # num_layer = 0
            # for name, param in model.named_parameters():
            #     if param.requires_grad and 'bias' not in name:
            #         grads_norms['grad_w' + str(100 + num_layer)[1:]] = \
            #             torch.norm(param.grad).item()
            #         num_layer += 1
            # log_writer.add_scalars('gradients_norms',
            #                        grads_norms,
            #                        n_iter_train)

            n_iter_train += 1

        # validating process
        model.eval()
        loss_val_batch = []
        metrics_val_batch = []
        for x, target, target_bin in tqdm(val_loader, desc='validation', position=0):
            with torch.no_grad():
                x = x.to(device=cuda, dtype=torch.float32, non_blocking=True)
                target = target.to(device=cuda, non_blocking=True)
                target_bin = target_bin.to(device=cuda, non_blocking=True)

                out = model.forward(x)
                loss = loss_fn(out, target_bin)

                # calc metrics
                max_out = torch.argmax(out, 1)
                bin_centroids = (torch_bins[max_out] + torch_bins[max_out + 1]) / 2
                bin_centroids = bin_centroids.to(device=cuda, non_blocking=True)
                metrics = F.l1_loss(bin_centroids, target)

                loss_val_batch.append(loss.item())
                metrics_val_batch.append(metrics.item())

        # change lr
        val_mean_loss = np.mean(loss_val_batch)
        val_mean_metrics = np.mean(metrics_val_batch)
        lr_scheduler.step(val_mean_metrics)

        # logging
        log_writer.add_scalars('loss',
                               {'val_mean': val_mean_loss},
                               n_iter_train - 1)
        log_writer.add_scalars('metrics_MAE',
                               {'val_mean': val_mean_metrics},
                               n_iter_train - 1)
        log_writer.add_scalar('lr',
                              optimizer.param_groups[0]['lr'],
                              n_iter_train - 1)

        # saving model TODO: make generator to save N last epochs
        save_path = os.path.join(logs_path, model_name + '_last_state.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mean_loss': val_mean_loss,
            'val_mean_metrics': val_mean_metrics
        }, save_path)

        if epoch == 0:
            best_val_mean_loss = val_mean_loss
        else:
            if val_mean_loss < best_val_mean_loss:
                best_val_mean_loss = val_mean_loss

                save_path = os.path.join(logs_path, model_name + '_best_state.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mean_loss': val_mean_loss,
                    'val_mean_metrics': val_mean_metrics
                }, save_path)
