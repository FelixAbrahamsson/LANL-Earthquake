import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
from scipy import stats
import numpy as np


class EarthquakeDatasetTrain(Dataset):

    def __init__(self, data, window_step=1, mask_prob=0.0):

        self.data = data
        self.window_size = 150000
        self.window_step = window_step
        self.n_data = len(self.data) // self.window_step - self.window_size // self.window_step
        self.mask_prob = mask_prob
        self.engineer = FeatureEngineer()

    def __len__(self):

        return self.n_data

    def __getitem__(self, idx):

        start = idx * self.window_step
        stop = start + self.window_size 
        slice_data = self.data[start : stop]

        seq = slice_data[:, 0]

        if self.mask_prob:
            seq_mask = np.random.random(len(seq)) < self.mask_prob
            seq[seq_mask] = np.nan

        features = self.engineer(seq)

        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(slice_data[-1, 1], dtype=torch.float)

        sample = {
            'features' : features,
            'labels' : label,
        }

        return sample


class EarthquakeDatasetTest(Dataset):

    def __init__(self, test_dir, scale_fnc=None):

        self.test_files = glob.glob(test_dir + '*')
        self.scale_fnc = scale_fnc
        self.engineer = FeatureEngineer()

    def __len__(self):

        return len(self.test_files)

    def __getitem__(self, idx):

        file_path = self.test_files[idx]
        seg_id = file_path.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(file_path)

        seq = df['acoustic_data'].values

        if scale_fnc is not None:
            seq = scale_fnc(seq)

        features = self.engineer(seq)

        features = torch.tensor(features, dtype=torch.float)

        sample = {
            'features' : features,
            'seg_id' : seg_id,
        }

        return sample


class FeatureEngineer():

    chunk_size = 1000
    n_features = 13

    def __call__(self, seq):

        n_chunks = len(seq) // self.chunk_size
        features = np.zeros((n_chunks, self.n_features))

        for i in range(n_chunks):

            chunk = seq[i * self.chunk_size : (i+1) * self.chunk_size]
            chunk = chunk[~np.isnan(chunk)]
            mean = chunk.mean()
            std = chunk.std()
            var = np.square(std)
            q1 = np.quantile(chunk, 0.25)
            q3 = np.quantile(chunk, 0.75)
            arg_sorted = chunk.argsort()
            top3 = arg_sorted[-3:]
            bottom3 = arg_sorted[:3]
            skew = stats.skew(chunk)
            kurt = stats.kurtosis(chunk)

            features[i, :] = np.array([
                mean,
                std,
                q1,
                q3,
                var,
                top3[0],
                top3[1],
                top3[2],
                bottom3[0],
                bottom3[1],
                bottom3[2],
                skew,
                kurt,
            ])

        return features