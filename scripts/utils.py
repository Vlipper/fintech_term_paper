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
    window = 'hann'  # {triang, hamming, hann}
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


def bin_to_target(net_out, bins):
    max_out = torch.argmax(net_out, 1)
    bin_centroids = (bins[max_out] + bins[max_out + 1]) / 2
    return bin_centroids


def left_padding(row, ref_len):
    pad_size = ref_len - len(row)
    pad = np.zeros(pad_size)
    row_padded = np.concatenate((pad, row))

    return row_padded


def apply_wd(model, gamma):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            continue
        tensor.data.add_(-gamma * tensor.data)


def calc_grad_norm(model):
    norm = 0.0
    count = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm += torch.norm(param.grad.data)
            count += 1
    if norm == 0 and count == 0:
        return 0
    else:
        return norm / count


def save_model(save_path, epoch, n_iter_train, model, optimizer,
               val_mean_loss, val_mean_metrics):
    torch.save({
        'epoch': epoch,
        'n_iter_train': n_iter_train,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mean_loss': val_mean_loss,
        'val_mean_metrics': val_mean_metrics
    }, save_path)


def train_clf_model(model, optimizer, lr_scheduler, train_loader, val_loader,
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
            log_writer.add_scalar('mean_grad_norm',
                                  calc_grad_norm(model),
                                  n_iter_train)

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
        val_mean_loss = np.mean(loss_val_batch)
        val_mean_metrics = np.mean(metrics_val_batch)

        # change lr
        lr_scheduler.step(val_mean_loss)

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
        save_model(save_path, epoch, model, optimizer, val_mean_loss, val_mean_metrics)

        if epoch == 0 or val_mean_metrics < best_val_mean_metrics:
            best_val_mean_metrics = val_mean_metrics

            save_path = os.path.join(logs_path, model_name + '_best_state.pth')
            save_model(save_path, epoch, model, optimizer,
                       val_mean_loss, val_mean_metrics)


def train_cpc_model(cpc_meta_model, train_loader, val_loader,
                    optimizer, lr_scheduler, num_epochs,
                    model_name, logs_path, log_writer, num_bins,
                    n_iter_train=1):
    cpc_meta_model.to(cuda)

    torch_bins = torch.linspace(0, 16.11, num_bins, dtype=torch.float32)
    # n_iter_train = 1
    for epoch in range(num_epochs):
        # training process
        cpc_meta_model.train()
        for large_x, target, target_bin in tqdm(train_loader,
                                                desc='training, epoch={}'.format(epoch + 1),
                                                position=0):
            large_x = large_x.to(device=cuda, non_blocking=True)
            target = target.to(device=cuda, non_blocking=True)
            target_bin = target_bin.to(device=cuda, non_blocking=True)
            optimizer.zero_grad()

            # calc logits from enc and AR models and target_bin_pred from
            # second head after AR model
            logits, target_bin_pred = cpc_meta_model.forward(large_x)

            # calc InfoNCE. Positives lies on diagonals.
            loss_cpc = - torch.diagonal(logits, dim1=-2, dim2=-1).mean()
            # permute target_bin_pred because shape must be (N, C, d1, ..., dn)
            loss_target = F.cross_entropy(target_bin_pred.permute(0, 2, 1), target_bin)
            # calc sum of losses and backward
            loss = loss_cpc + 0.05*loss_target
            loss.backward()
            optimizer.step()

            # calc metrics
            max_out = torch.argmax(target_bin_pred, -1)
            bin_centroids = (torch_bins[max_out] + torch_bins[max_out + 1]) / 2
            bin_centroids = bin_centroids.to(device=cuda, non_blocking=True)
            metrics = F.l1_loss(bin_centroids, target)

            # write logs
            log_writer.add_scalar('losses/train_sum', loss.item(), n_iter_train)
            log_writer.add_scalar('losses/train_cpc', loss_cpc.item(), n_iter_train)
            log_writer.add_scalar('losses/train_target', loss_target.item(), n_iter_train)
            log_writer.add_scalar('metrics_MAE/train', metrics.item(), n_iter_train)
            log_writer.add_scalar('grad_norms/encoder', calc_grad_norm(cpc_meta_model.encoder), n_iter_train)
            log_writer.add_scalar('grad_norms/ar', calc_grad_norm(cpc_meta_model.ar), n_iter_train)
            log_writer.add_scalar('grad_norms/target_head', calc_grad_norm(cpc_meta_model.target_head), n_iter_train)

            n_iter_train += 1

        # validating process
        cpc_meta_model.eval()
        loss_val_batch = [[], [], []]
        metrics_val_batch = []
        for large_x, target, target_bin in tqdm(val_loader, desc='validation', position=0):
            with torch.no_grad():
                large_x = large_x.to(device=cuda, non_blocking=True)
                target = target.to(device=cuda, non_blocking=True)
                target_bin = target_bin.to(device=cuda, non_blocking=True)

                # calc logits from enc and AR models and target_bin_pred from
                # second head after AR model
                logits, target_bin_pred = cpc_meta_model.forward(large_x)

                # calc InfoNCE. Positives lies on diagonals.
                loss_cpc = - torch.diagonal(logits, dim1=-2, dim2=-1).mean()
                # permute target_bin_pred because shape must be (N, C, d1, ..., dn)
                loss_target = F.cross_entropy(target_bin_pred.permute(0, 2, 1), target_bin)
                # calc sum of losses and backward
                loss = loss_cpc + 0.05*loss_target

                # calc metrics
                max_out = torch.argmax(target_bin_pred, -1)
                bin_centroids = (torch_bins[max_out] + torch_bins[max_out + 1]) / 2
                bin_centroids = bin_centroids.to(device=cuda, non_blocking=True)
                metrics = F.l1_loss(bin_centroids, target)

                loss_val_batch[0].append(loss.item())
                loss_val_batch[1].append(loss_cpc.item())
                loss_val_batch[2].append(loss_target.item())
                metrics_val_batch.append(metrics.item())
        val_mean_loss = np.mean(loss_val_batch[0])
        val_mean_metrics = np.mean(metrics_val_batch)

        # change lr
        lr_scheduler.step(val_mean_loss)

        # logging
        log_writer.add_scalar('losses/val_sum', val_mean_loss, n_iter_train - 1)
        log_writer.add_scalar('losses/val_cpc', np.mean(loss_val_batch[1]), n_iter_train - 1)
        log_writer.add_scalar('losses/val_target', np.mean(loss_val_batch[2]), n_iter_train - 1)
        log_writer.add_scalar('metrics_MAE/val', val_mean_metrics, n_iter_train - 1)
        log_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_iter_train - 1)

        # saving model
        save_path = os.path.join(logs_path, model_name + '_last_state.pth')
        save_model(save_path, epoch, n_iter_train, cpc_meta_model, optimizer,
                   val_mean_loss, val_mean_metrics)

        if epoch == 0 or val_mean_loss < best_val_mean_loss:
            best_val_mean_loss = val_mean_loss

            save_path = os.path.join(logs_path, model_name + '_best_state.pth')
            save_model(save_path, epoch, n_iter_train, cpc_meta_model, optimizer,
                       val_mean_loss, val_mean_metrics)
