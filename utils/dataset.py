import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import glob
import pandas as pd
from scipy import stats
import numpy as np


class EarthquakeDatasetTrain(Dataset):

    def __init__(self, data, window_step=1, mask_prob=0.0):

        self.data = data
        self.window_size = 150000
        self.window_step = window_step
        self.n_data = (len(self.data) - self.window_size) // self.window_step
        self.mask_prob = mask_prob

    def __len__(self):

        return self.n_data

    def __getitem__(self, idx):

        start = idx * self.window_step
        stop = start + self.window_size 
        slice_data = self.data[start : stop]

        seq = slice_data[:, 0]
        label = slice_data[-1, 1]

        if self.mask_prob:
            seq_mask = np.random.random(len(seq)) < self.mask_prob
            seq[seq_mask] = np.nan

        features = engineer_features(seq)

        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        sample = {
            'features' : features,
            'labels' : label,
        }

        return sample


class EarthquakeDatasetTest(Dataset):

    def __init__(self, test_dir, scale_fnc=None):

        self.test_files = glob.glob(test_dir + '*')
        self.scale_fnc = scale_fnc

    def __len__(self):

        return len(self.test_files)

    def __getitem__(self, idx):

        file_path = self.test_files[idx]
        seg_id = file_path.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(file_path)

        seq = df['acoustic_data'].values

        if scale_fnc is not None:
            seq = scale_fnc(seq)

        features = engineer_features(seq)

        features = torch.tensor(features, dtype=torch.float)

        sample = {
            'features' : features,
            'seg_id' : seg_id,
        }

        return sample


class RandomLoader:

    def __init__(self, dataset, batch_size=1, num_epoch_steps=1000):

        sampler = RandomSampler(dataset, 
            replacement=True, 
            num_samples=batch_size)

        self.data_loader = DataLoader(dataset, 
                          sampler=sampler,
                          batch_size=batch_size)
        self.data_iter = iter(self.data_loader)
        self.dataset = dataset
        self.num_epoch_steps = num_epoch_steps

    def __len__(self):
        return self.num_epoch_steps

    def __iter__(self):

        for step in range(self.num_epoch_steps):
            try:
                yield next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.data_loader)
                yield next(self.data_iter)


def engineer_features(seq, chunk_size=1000):

    n_chunks = len(seq) // chunk_size
    features = []

    for i in range(n_chunks):

        chunk = seq[i * chunk_size : (i+1) * chunk_size]

        chunk_features = process_chunk(chunk)
        chunk_features.extend(process_chunk(chunk[-100:]))
        chunk_features.extend(process_chunk(chunk[-10:]))
        
        features.append(chunk_features)

    return np.array(features)


def process_chunk(chunk):

        mean = chunk.mean()
        std = chunk.std()
        # var = np.square(std)
        # q1 = np.quantile(chunk, 0.25)
        # q3 = np.quantile(chunk, 0.75)
        # arg_sorted = chunk.argsort()
        # top3 = arg_sorted[-3:]
        # bottom3 = arg_sorted[:3]
        # skew = stats.skew(chunk)
        # kurt = stats.kurtosis(chunk)

        minimum = chunk.min()
        maximum = chunk.max()

        features = [
            mean,
            std,
            # q1,
            # q3,
            # var,
            # top3[0],
            # top3[1],
            # top3[2],
            # bottom3[0],
            # bottom3[1],
            # bottom3[2],
            # skew,
            # kurt,
            minimum,
            maximum
        ]

        return features