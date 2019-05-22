import os
import subprocess
import time
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
            x = x.to(device=cuda, dtype=torch.float32, non_blocking=True)

            out = model.forward(x)
            out = out.squeeze().to(cpu).tolist()
            if isinstance(out, list):
                preds.extend(out)
            elif isinstance(out, float):
                preds.append(out)
            else:
                raise Exception('out is not float or list')
    return preds


num_bins = 17

model = tv_models.resnet34(pretrained=False)
model = models.get_resnet(model, out_size=num_bins-1)
model_name = 'spectr_net_v1_best_state.pth'  # _best_state, _last_state

# model_path = '/mntlong/lanl_comp/logs/' + model_name
file_dir = os.path.dirname(__file__)
logs_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'logs'))
model_path = os.path.join(logs_path, model_name)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model = model.to(cuda)
model.eval()

# test_data_path = '/mntlong/lanl_comp/data/test/'
data_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'data'))
test_data_path = os.path.join(data_path, 'test')
test_names = os.listdir(test_data_path)
# test_names = test_names[:50]

batch_size = 1000  # 1300

hz_cutoff = 600000  # {0, ..., 600000, ...}
window_size = 150000
overlap_size = int(window_size * 0.0)
nperseg = 2048

for wave_num, test_wave in enumerate(tqdm(test_names,
                                          desc='test inference',
                                          position=0)):
    wave_data = np.loadtxt(os.path.join(test_data_path, test_wave),
                           dtype=np.float32, skiprows=1)
    test_dataset = data.SpectrogramDataset(wave_data, target=None,
                                           idxs_wave_end=[1500000],
                                           num_bins=None,
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
    # print('out preds size', preds.size())
    # preds = np.argmax(preds, 1)
    # print('argmax out preds size', preds.size())
    # print('---------------------')

    if wave_num == 0:
        preds_mtrx = np.array([]).reshape(0, len(preds))

    preds_mtrx = np.vstack((preds_mtrx, preds))

max_preds = np.argmax(preds_mtrx, 1)
bins = np.linspace(0, 16.11, num_bins)
bin_centroids = (bins[max_preds] + bins[max_preds + 1]) / 2

# print('preds_mtrx size, mean and std: {}, {:.4f}, {:.4f}'
#       .format(preds_mtrx.shape, np.mean(preds_mtrx), np.std(preds_mtrx)))

# preds_out = preds_mtrx[:, -1]
preds_out = bin_centroids

submit = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))  # , nrows=50
submit.loc[:, 'time_to_failure'] = preds_out

subs_path = os.path.join(data_path, 'subs')
try:
    os.mkdir(subs_path)
except FileExistsError:
    pass
submit_path = os.path.join(subs_path, 'sub_' + model_name[:-15] + '.csv',)
submit.to_csv(submit_path, index=False)

# submit to kaggle
submit_command = "kaggle competitions submit -c LANL-Earthquake-Prediction " \
                 "-f {} -m 'spect_resnet34'".format(submit_path)
if subprocess.run(submit_command, shell=True).returncode == 0:
    print('waiting 20 sec. for results')
    time.sleep(20)
else:
    raise Exception('submit was not done')

print('Submissions info')
subprocess.run('kaggle competitions submissions -c LANL-Earthquake-Prediction',
               shell=True)
