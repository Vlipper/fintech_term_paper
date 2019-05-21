# import numpy as np
# import models
#
# # train_signal = torch.from_numpy(train_signal)
# # train_quaketime = torch.from_numpy(train_quaketime)
#
#
# train_signal = np.load('../data/train_compressed.npz')['signal']
# train_signal = np.array(train_signal)
#
# print(train_signal.shape)
#
#
# #%% tests
# import torch
# inpt = torch.rand(3, 1, 150000)
# model = BaselineNetRawSignalCnnRnnV1()
# out = model(inpt)
# out.size()
#
#
# #%% end_tests
#
# from itertools import islice
# import numpy as np
#
# tst = islice(np.arange(10), 2)
#
#
# ## torchaudio paddings
# padding_size = [self.max_len - tensor.size(self.len_dim) if i == self.len_dim
#                 else tensor.size(self.ch_dim)
#                 for i in range(2)]
# pad = torch.empty(padding_size, dtype=tensor.dtype).fill_(self.fill_value)
# tensor = torch.cat((tensor, pad), dim=self.len_dim)

import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import utils


def main():
    inpt = np.random.randint(-10, 10, 150000)

    for i in range(1000):
        out = utils.spectrogram(inpt, 256)


if __name__ == '__main__':
    main()
