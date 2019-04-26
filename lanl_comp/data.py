# import numpy as np
# import pandas as pd
import torch
from torch.utils.data import Dataset
import utils


# signal dataset
class SignalDataset(Dataset):
    def __init__(self, signal, target, exmpl_size=10000):
        super().__init__()

        self.signal = signal
        self.target = target
        # make borders for every example
        self.boarder_points = [(i * exmpl_size, (i + 1) * exmpl_size)
                               for i in range(signal.shape[0] // exmpl_size + 1)]
        self.boarder_points = self.boarder_points[:-1]

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        return self.signal[start_idx:end_idx].view(1, -1), self.target[end_idx - 1]


# spectrogram dataset
class SpectrogramDataset(Dataset):
    def __init__(self, signal, target=None, hz_cutoff=600000,
                 window_size=10000, overlap_size=5000):
        super().__init__()

        self.signal = signal
        self.target = target
        self.hz_cutoff = hz_cutoff

        # make borders for every example
        # self.boarder_points = [(i * exmpl_size, (i + 1) * exmpl_size)
        #                        for i in range(signal.shape[0] // exmpl_size + 1)]
        # self.boarder_points = self.boarder_points[:-1]
        wave_size = signal.shape[0]
        self.boarder_points = [(i, i + window_size)
                               for i in range(0, wave_size, window_size - overlap_size)
                               if i + window_size <= wave_size]

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        f, _, spec = utils.spectrogram(self.signal[start_idx:end_idx])

        if self.hz_cutoff:
            cutoff = f[f <= self.hz_cutoff].shape[0]
            spec = torch.from_numpy(spec[:cutoff, :])
        else:
            spec = torch.from_numpy(spec)

        if self.target is not None:
            target = torch.tensor(self.target[end_idx - 1])
            return spec.view(1, spec.size(0), -1), target
        else:
            return spec.view(1, spec.size(0), -1)
