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
    window = 'hamming'  # {triang, hamming}
    detrend = 'linear'  # {'linear', 'constant', False}
    scaling = 'density'  # {'density', 'spectrum'}
    eps = 1e-11
    f, t, Sxx = signal.spectrogram(sig_in, fs=fs, window=window,
                                   nperseg=nperseg, noverlap=noverlap,
                                   detrend=detrend, scaling=scaling)
    return f, t, np.log(Sxx + eps)


def train_model(model, optimizer, lr_scheduler, train_loader, val_loader,
                num_epochs, model_name, logs_path, log_writer):  # loss_fn,
    model = model.to(cuda)
    n_iter_train = 1
    for epoch in range(num_epochs):
        # training process
        model.train()
        for x, target in tqdm(train_loader,
                              desc='training, epoch={}'.format(epoch + 1),
                              position=0):
            x = x.to(cuda, non_blocking=True).float()
            target = target.to(cuda, non_blocking=True)

            optimizer.zero_grad()
            out = model.forward(x)
            loss = F.l1_loss(out, target)

            loss.backward()
            optimizer.step()

            # logging
            log_writer.add_scalars('loss',
                                   {'train_batch': loss.item()},
                                   n_iter_train)
            n_iter_train += 1

        # validating process
        loss_val_batch = []
        model.eval()

        # calculating loss
        for x, target in tqdm(val_loader, desc='validation', position=0):
            with torch.no_grad():
                x = x.to(cuda, non_blocking=True).float()
                target = target.to(cuda, non_blocking=True)

                out = model.forward(x)
                loss = F.l1_loss(out, target)

                loss_val_batch.append(loss.item())

        # change lr
        val_mean_loss = np.mean(loss_val_batch)
        lr_scheduler.step(val_mean_loss)

        # logging
        log_writer.add_scalars('loss',
                               {'val_mean': val_mean_loss},
                               n_iter_train - 1)
        log_writer.add_scalar('lr',
                              optimizer.param_groups[0]['lr'],
                              n_iter_train - 1)

        # saving model (make generator to save N last epochs)
        save_path = logs_path + model_name + '_last_state.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mean_loss': val_mean_loss,
            # 'val_metrics': val_mean_roc_metrics
        }, save_path)

        if epoch == 0:
            best_val_mean_loss = val_mean_loss
        else:
            if val_mean_loss < best_val_mean_loss:
                best_val_mean_loss = val_mean_loss

                save_path = logs_path + model_name + '_best_state.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mean_loss': val_mean_loss,
                    # 'val_metrics': val_mean_roc_metrics
                }, save_path)
