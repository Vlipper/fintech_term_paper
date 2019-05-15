import os
from tqdm import tqdm

import numpy as np
import pandas as pd

# import local scripts
import data
import models

import torch
from torchvision import models as tv_models
from torch.utils.data import DataLoader
cuda = torch.device('cuda')
cpu = torch.device('cpu')


def test_inference(model, data_loader):
    preds = []
    for x in data_loader:  # tqdm(data_loader, desc='test inference', position=0):
        with torch.no_grad():
            x = x.to(cuda, non_blocking=True).float()

            out = model.forward(x)
            out = out.squeeze().to(cpu).tolist()
            if isinstance(out, list):
                preds.extend(out)
            elif isinstance(out, float):
                preds.append(out)
            else:
                raise Exception('out is not float or list')
    return preds


model = tv_models.resnet50(pretrained=False)
model = models.get_resnet(model)
model_name = 'wave_net_v1_new3_best_state.pth'  # _best_state, _last_state

model_path = '/mntlong/scripts/logs/' + model_name
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model = model.to(cuda)
model.eval()

data_path = '/mntlong/scripts/data/'
test_data_path = data_path + 'test/'
test_names = os.listdir(test_data_path)
test_names = test_names[:50]

batch_size = 100  # 1300

hz_cutoff = 600000  # {0, ..., 600000, ...}
window_size = 150000
overlap_size = int(window_size * 0.0)
nperseg = 2048

for wave_num, test_wave in enumerate(tqdm(test_names,
                                          desc='test inference',
                                          position=0)):
    wave_data = np.loadtxt(test_data_path + test_wave,
                           dtype=np.float32, skiprows=1)
    test_dataset = data.SpectrogramDataset(wave_data, target=None,
                                           hz_cutoff=hz_cutoff,
                                           window_size=window_size,
                                           overlap_size=overlap_size,
                                           nperseg=nperseg)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=5,
                             pin_memory=True)
    preds = test_inference(model, test_loader)

    if wave_num == 0:
        preds_mtrx = np.array([]).reshape(0, len(preds))
    preds_mtrx = np.vstack((preds_mtrx, preds))

print('preds_mtrx mean and std: {:.4f}, {:.4f}'\
      .format(np.mean(preds_mtrx), np.std(preds_mtrx)))

preds_out = preds_mtrx[:, -1]

submit = pd.read_csv(data_path + 'sample_submission.csv')
submit.loc[:, 'time_to_failure'] = preds_out
submit.to_csv('sub_' + model_name[:-15] + '.csv', index=False)
