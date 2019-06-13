import numpy as np
import torch
from torch.utils.data import Dataset
import utils


# signal dataset
class SignalDataset(Dataset):
    def __init__(self, signal, target, idxs_wave_end, num_bins,
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

        # split target on bins
        if target is not None:
            bins = np.linspace(0, 16.11, num_bins)
            self.target_bins = np.digitize(target, bins) - 1

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        signal = torch.from_numpy(self.signal[start_idx:end_idx])

        signal = (signal - 4.5195) / 10.7357

        if self.target is not None:
            target = torch.tensor(self.target[end_idx - 1])
            target_bin = torch.tensor(self.target_bins[end_idx - 1])
            return signal.view(1, -1), target, target_bin
        else:
            return signal.view(1, -1)


# spectrogram dataset
class SpectrogramDataset(Dataset):
    def __init__(self, signal, target, idxs_wave_end, num_bins, stdscale=True,
                 hz_cutoff=600000, window_size=10000, overlap_size=5000, nperseg=256):
        super().__init__()

        self.signal = signal
        self.target = target
        self.hz_cutoff = hz_cutoff
        self.stdscale = stdscale

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
            spec = spec[:cutoff, :]
            spec = torch.from_numpy(spec)
        else:
            spec = torch.from_numpy(spec)

        if self.stdscale:
            spec = (spec - (-12)) / 1.7
            # spec = torch.clamp(spec, (-12) - 2*1.7, (-12) + 2*1.7)

        if self.target is not None:
            target = torch.tensor(self.target[end_idx - 1])
            target_bin = torch.tensor(self.target_bins[end_idx - 1])
            return spec.view(1, spec.size(0), -1), target, target_bin
        else:
            return spec.view(1, spec.size(0), -1)


# signal CPC dataset
class SignalCPCDataset(Dataset):
    def __init__(self, signal, target, idxs_wave_end, num_bins,
                 large_ws=150000, overlap_size=5000, small_ws=15000,
                 scale_clamp=True, clamp_min=-10, clamp_max=10):
        super().__init__()

        self.signal = signal
        self.target = target
        self.small_ws = small_ws
        self.scale_clamp = scale_clamp
        self.mean = 4.5195
        self.std = 10.7357
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # make borders for every example
        #  and exclude borders which include different waves
        next_wave_windows = [(i, i + large_ws) for i in idxs_wave_end]
        wave_size = signal.shape[0]
        self.boarder_points = [(i, i + large_ws)
                               for i in range(0, wave_size,
                                              large_ws - overlap_size)
                               if i + large_ws <= wave_size
                               and utils.other_wave_check(i + large_ws,
                                                          next_wave_windows)]

        # split target on bins
        if target is not None:
            bins = np.linspace(0, 16.11, num_bins)
            self.target_bins = np.digitize(target, bins) - 1

    def __len__(self):
        return len(self.boarder_points)

    def __getitem__(self, index):
        start_idx, end_idx = self.boarder_points[index]
        signal = self.signal[start_idx:end_idx]

        if self.scale_clamp:
            signal = (signal - self.mean) / self.std
            signal = np.clip(signal, self.clamp_min, self.clamp_max)

        # if signal.shape[0] < self.small_ws:
        #     signal = utils.left_padding(signal, self.small_ws)

        signal = torch.from_numpy(signal)
        signal = signal.view(-1, 1, self.small_ws)  # size is (bs_x, 1, small_ws)
        # signal = signal.view(1, -1)

        if self.target is not None:
            target = np.append(0, self.target[start_idx:end_idx])[::self.small_ws][1:]
            target_bin = np.append(0, self.target_bins[start_idx:end_idx])[::self.small_ws][1:]
            target, target_bin = torch.from_numpy(target), torch.from_numpy(target_bin)

            return signal.float(), target.float(), target_bin
        else:
            return signal.float()
