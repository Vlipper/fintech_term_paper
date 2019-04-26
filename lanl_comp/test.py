import os
from tqdm import tqdm

import numpy as np
import pandas as pd

# import local scripts
import data
import models

import torch
from torch.utils.data import DataLoader
cuda = torch.device('cuda')
cpu = torch.device('cpu')


def test_inference(model, data_loader):
    preds = []
    for x in data_loader:  # tqdm(data_loader, desc='test inference', position=0):
        with torch.no_grad():
            x = x.to(cuda, non_blocking=True).float()

            out = model.forward(x)
            preds.extend(out.squeeze().to(cpu).tolist())
    return preds


model = models.BaselineNetSpect()
model_name = 'spectr_net_v0_last_state.pth'

model_path = '/mntlong/lanl_comp/logs/' + model_name
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model = model.to(cuda)
model.eval()

data_path = '/mntlong/lanl_comp/data/'
test_data_path = data_path + 'test/'
test_names = os.listdir(test_data_path)

batch_size = 1000
window_size = 10000

for wave_num, test_wave in enumerate(tqdm(test_names,
                                          desc='test inference',
                                          position=0)):
    wave_data = np.loadtxt(test_data_path + test_wave,
                           dtype=np.float32, skiprows=1)
    test_loader = DataLoader(
        dataset=data.SpectrogramDataset(wave_data, target=None,
                                        hz_cutoff=0, exmpl_size=window_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True)
    preds = test_inference(model, test_loader)

    if wave_num == 0:
        preds_mtrx = np.array([]).reshape(0, len(preds))
    preds_mtrx = np.vstack((preds_mtrx, preds))

    # if wave_num == 5:
    #     break

preds_out = preds_mtrx[:, -1]

print('preds mean and std: {:.4f}, {:.4f}'\
      .format(np.mean(preds), np.std(preds)))

submit = pd.read_csv(data_path + 'sample_submission.csv')
submit.loc[:, 'time_to_failure'] = preds_out
submit.to_csv('sub_' + model_name[:-15] + '.csv', index=False)
