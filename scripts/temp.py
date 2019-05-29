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
#
# train_data_path = os.path.join(data_path, 'train.csv')
# train_data = pd.read_csv(train_data_path,
#                          dtype={'acoustic_data': np.int16,
#                                 'time_to_failure': np.float32})
# train_signal = train_data['acoustic_data'].values
# train_quaketime = train_data['time_to_failure'].values
# del train_data
#
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

# import os
# os.environ['MKL_NUM_THREADS'] = '1'
# import numpy as np
# import utils
#
#
# def main():
#     inpt = np.random.randint(-10, 10, 150000)
#
#     for i in range(1000):
#         out = utils.spectrogram(inpt, 256)
#
#
# if __name__ == '__main__':
#     main()

from lr_finder import LRFinder

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=1, num_iter=50, step_mode="exp")
lr_finder.get_best_lr()
# lr_finder.plot()
# lr_finder.history


# import argparse
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--find_lr', default=False, action='store_true')
#     parser.add_argument('--model_name', default='spectr_net_default')
#     parser.add_argument('--num_epochs', default=10)
#     parser.add_argument('--batch_size', default=120)
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     print(type(args.find_lr), args.find_lr)
