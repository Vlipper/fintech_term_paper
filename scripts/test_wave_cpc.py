import os
from tqdm import tqdm
import subprocess
import time

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
    for x in data_loader:
        with torch.no_grad():
            x = x.to(device=cuda, non_blocking=True)

            _, out = model.forward(x)
            out = out.squeeze().to(cpu).tolist()
            if isinstance(out, list):
                preds.extend(out)
            elif isinstance(out, float):
                preds.append(out)
            else:
                raise Exception('out is not float or list')
    return preds


def main():
    num_bins = 17

    model = models.CPCv1(out_size=num_bins-1)
    model_name = 'wave_net_cpc_default' + '_best_state.pth'  # _best_state, _last_state

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
    # test_names = test_names[:200]

    batch_size = 10

    # params
    large_ws = 150000
    overlap_size = int(large_ws * 0.0)
    small_ws = 150000

    for wave_num, test_wave in enumerate(tqdm(test_names,
                                              desc='test inference',
                                              position=0)):
        wave_data = np.loadtxt(os.path.join(test_data_path, test_wave),
                               dtype=np.float32, skiprows=1)

        test_dataset = data.SignalCPCDataset(wave_data, target=None, num_bins=None,
                                             idxs_wave_end=[1500000],
                                             large_ws=large_ws,
                                             overlap_size=overlap_size,
                                             small_ws=small_ws)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=5,
                                 pin_memory=True)
        preds = test_inference(model, test_loader)

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
                     "-f {} -m 'test'".format(submit_path)
    if subprocess.run(submit_command, shell=True).returncode == 0:
        print('\n', 'wait 20 sec. for results')
        time.sleep(10)
    else:
        raise Exception('submit was not done')

    print('Submissions info')
    subprocess.run('kaggle competitions submissions -c LANL-Earthquake-Prediction', shell=True)


if __name__ == '__main__':
    main()
