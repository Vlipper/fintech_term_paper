import numpy as np
import models


train_signal = np.load('../data/train_compressed.npz')['signal']
train_signal = np.array(train_signal)

print(train_signal.shape)


#%% tests
import torch
inpt = torch.rand(3, 1, 150000)
model = BaselineNetRawSignalCnnRnnV1()
out = model(inpt)
out.size()


#%% end_tests



# Dataset
# class SignalTestDataset(Dataset):
#     def __init__(self, files_path, file_names_list):
#         super().__init__()
#
#         self.files_path = files_path
#         self.file_names_list = file_names_list
#
#     def __len__(self):
#         return len(self.file_names_list)
#
#     def __getitem__(self, index):
#         file = np.loadtxt(self.files_path + self.file_names_list[index],
#                           dtype=np.float32,
#                           skiprows=1 + 140000)  # skiprows temp decision
#         return torch.from_numpy(file).view(1, -1)

window_size = 100
wave_size = 1000
overlap_size = window_size // 2
# n = 3  # group size
# m = 1  # overlap size

# n = 1
# tst = [(i * window_size + n, (i + 1) * window_size + n)
#        for i in range(wave_size // window_size + 1)]

tst = [(i, i + window_size)
       for i in range(0, wave_size, window_size - overlap_size)
       if i + window_size <= wave_size]
tst


from itertools import islice
import numpy as np

tst = islice(np.arange(10), 2)
