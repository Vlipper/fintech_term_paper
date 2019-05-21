import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# data_path = './data/train/train.csv'
file_dir = os.path.dirname(__file__)
data_path = os.path.abspath(os.path.join(file_dir, os.path.pardir, 'data'))

# load source csv to pandas
chunks = []
for chunk in tqdm(pd.read_csv(os.path.join(data_path, 'train.csv'),
                              chunksize=10 ** 6, low_memory=False,  # nrows=10**7 * 30,
                              dtype={'acoustic_data': np.int16,
                              'time_to_failure': np.float32})):  # np.float64
    chunks.append(chunk)
train_source = pd.concat(chunks)
train_source.columns = ['signal', 'quake_time']
del chunks

# определяем индексы начала экспириментов
quake_time = train_source['quake_time'].values
index_start = np.nonzero(np.diff(quake_time) > 0)[0] + 1
index_start = np.insert(index_start, 0, 0)

# сохраняем индексы начала и конца экспириментов
train_info = pd.DataFrame({
    'indx_start': index_start,
    'indx_end': np.append(index_start[1:], quake_time.shape[0])
})
train_info.to_csv(os.path.join(data_path, 'train_info.csv'))

# save compressed source data to npz
np.savez_compressed(os.path.join(data_path, 'train_data.npz'),
                    signal=train_source['signal'].values,
                    quake_time=train_source['quake_time'].values)
