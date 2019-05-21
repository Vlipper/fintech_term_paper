import numpy as np
import torch
from torch.utils.data import Dataset
import utils


# signal dataset
class SignalDataset(Dataset):
    def __init__(self, signal, target, idxs_wave_end,
                 window_size=10000, overlap_size=5000):
        super().__init__()

        self.signal = signal
        self.target = target

        # make borders for every example
        #  and exclude borders which include different waves
        next_wave_windows = [(i, i + window_size) for i in idxs_wave_end]
        wave_size = signal.shape[0]
        self.boarder_points = [(i, i + window_size)
                               for i in range(0, wave_size,
                                              window_size - overlap_size)
                               if i + window_size <= wave_size
                               and utils.other_wave_check(i + window_size,
                                                          next_wave_windows)]

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        signal = torch.from_numpy(self.signal[start_idx:end_idx])

        if self.target is not None:
            target = torch.tensor(self.target[end_idx - 1])
            return signal.view(1, -1), target
        else:
            return signal.view(1, -1)


# spectrogram dataset
class SpectrogramDataset(Dataset):
    def __init__(self, signal, target, idxs_wave_end, num_bins,
                 hz_cutoff=600000, window_size=10000, overlap_size=5000, nperseg=256):
        super().__init__()

        self.signal = signal
        self.target = target
        self.hz_cutoff = hz_cutoff

        # spectrogram func params
        self.nperseg = nperseg

        # make borders for every example
        #  and exclude borders which include different waves
        next_wave_windows = [(i, i + window_size) for i in idxs_wave_end]
        wave_size = signal.shape[0]
        self.boarder_points = [(i, i + window_size)
                               for i in range(0, wave_size, window_size - overlap_size)
                               if i + window_size <= wave_size
                               and utils.other_wave_check(i + window_size,
                                                          next_wave_windows)]

        # split target on bins
        if target is not None:
            bins = np.linspace(0, 16.11, num_bins)
            self.target_bins = np.digitize(target, bins) - 1

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        f, _, spec = utils.spectrogram(self.signal[start_idx:end_idx],
                                       self.nperseg)

        if self.hz_cutoff:
            cutoff = f[f <= self.hz_cutoff].shape[0]
            spec = torch.from_numpy(spec[:cutoff, :])
        else:
            spec = torch.from_numpy(spec)

        if self.target is not None:
            target = torch.tensor(self.target[end_idx - 1])
            target_bin = torch.tensor(self.target_bins[end_idx - 1])
            return spec.view(1, spec.size(0), -1), target, target_bin
        else:
            return spec.view(1, spec.size(0), -1)
