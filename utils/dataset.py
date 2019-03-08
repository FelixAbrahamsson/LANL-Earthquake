import torch
from torch.utils.data import Dataset
import glob
import pandas as pd


class EarthquakeDatasetPieces(Dataset):

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.n_data = len(glob.glob(self.data_dir))

    def __len__(self):

        return self.n_data

    def __getitem__(self, idx):

        df = pd.read_csv(self.data_dir + 'train_{}.csv'.format(idx))

        features = torch.tensor(df['acoustic_data'].values, dtype=torch.float)
        label = torch.tensor(df['time_to_failure'].values[-1], dtype=torch.float)

        sample = {
            'features' : features,
            'labels' : label,
        }

        return sample


class EarthquakeDatasetFull(Dataset):

    def __init__(self, data, window_step=1, mask_prob=0.0):

        self.data = data
        self.window_size = 150000
        self.window_step = window_step
        self.n_data = len(self.data) // self.window_step - self.window_size // self.window_step
        self.mask_prob = mask_prob

    def __len__(self):

        return self.n_data

    def __getitem__(self, idx):

        start = idx * self.window_step
        stop = start + self.window_size 
        slice_data = self.data[start : stop]

        features = torch.tensor(slice_data[:, 0], dtype=torch.float)
        label = torch.tensor(slice_data[-1, 1], dtype=torch.float)

        if self.mask_prob:
            feature_mask = (torch.rand(len(features)) < self.mask_prob).float()
            features = features * feature_mask

        sample = {
            'features' : features,
            'labels' : label,
        }

        return sample


class EarthquakeDatasetTest(Dataset):

    def __init__(self, test_dir):

        self.test_files = glob.glob(test_dir + '*')

    def __len__(self):

        return len(self.test_files)

    def __getitem__(self, idx):

        file_path = self.test_files[idx]
        seg_id = file_path.split('/')[-1].split('.csv')[0]
        df = pd.read_csv(file_path)

        features = torch.tensor(df['acoustic_data'].values, dtype=torch.float)

        sample = {
            'features' : features,
            'seg_id' : seg_id,
        }

        return sample