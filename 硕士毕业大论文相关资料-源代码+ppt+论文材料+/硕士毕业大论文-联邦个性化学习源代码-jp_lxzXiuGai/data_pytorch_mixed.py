import os
import numpy as np

from data import data as data_dct

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MixedLayoutHotspotDataset(Dataset):
    '''EDA layout hotspot dataset.'''

    def __init__(self, benchmark_paths: list, transform='train', resample=True):
        assert len(benchmark_paths) > 0

        self.resample = resample
        self.transform = transform
        self.feature_shape = (12, 12, -1)   # '-1' should be 32 channels

        label_files = []
        data_raws = []
        for b_path in benchmark_paths:
            _label_file = os.path.join(b_path, 'label.csv')
            label_files.append(_label_file)
            data_raws.append(data_dct(b_path, _label_file, preload=True))

        dataset_names = [b_path.split('/')[-1] for b_path in benchmark_paths]
        features = [d.ft_buffer for d in data_raws]
        labels = [d.label_buffer for d in data_raws]
        self.n_dataset = len(labels)
        del data_raws, label_files

        self._make_balanced_datasets(features, labels, dataset_names)
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.long)

        self.transform = transforms.ToTensor()

        

    def _make_balanced_datasets(self, features, labels, dataset_names):
        '''Balance each dataset'''
        if self.resample and self.transform != 'test':
            for i in range(self.n_dataset):
                ft = features[i]
                lb = labels[i]
                pos_idx = lb == 1
                neg_idx = lb == 0
                if pos_idx.sum() < neg_idx.sum():
                    resampled_ft = np.stack(ft[pos_idx])
                    resampled_lb = np.stack(lb[pos_idx])
                    n_repeat = neg_idx.sum() // pos_idx.sum() - 1
                    resampled_ft = np.tile(resampled_ft, (n_repeat, 1, 1))
                    resampled_lb = np.tile(resampled_lb, (n_repeat,))
                    ft = np.vstack(
                        (ft, resampled_ft))
                    lb = np.hstack(
                        (lb, resampled_lb))
                else:
                    resampled_ft = np.stack(ft[neg_idx])
                    resampled_lb = np.stack(lb[neg_idx])
                    n_repeat = pos_idx.sum() // neg_idx.sum() - 1
                    resampled_ft = np.tile(resampled_ft, (n_repeat, 1, 1))
                    resampled_lb = np.tile(resampled_lb, (n_repeat))
                    ft = np.vstack(
                        (ft, resampled_ft))
                    lb = np.hstack(
                        (lb, resampled_lb))
                features[i] = ft
                labels[i] = lb
                print("Resampled dataset {} to size {}".format(
                    dataset_names[i], ft.shape))
        '''Balance between datasets'''
        max_samples = max([len(lb) for lb in labels])
        for i in range(self.n_dataset):
            this_samples = len(labels[i])
            if this_samples < max_samples:
                n_repeat, n_remainder = divmod(max_samples, this_samples)
                features[i] = np.tile(features[i], (n_repeat, 1, 1))
                labels[i] = np.tile(labels[i], (n_repeat,))
                idx_remainder = np.random.choice(
                    len(labels[i]), n_remainder, replace=False)
                features[i] = np.vstack(
                    (features[i], features[i][idx_remainder]))
                labels[i] = np.hstack((labels[i], labels[i][idx_remainder]))
                print('Expand {} dataset size to {}'.format(
                    dataset_names[i], features[i].shape))

        for i in range(self.n_dataset):
            _name = dataset_names[i].split('_')[0]
            _mean = np.load('npy/{}-mean.npy'.format(_name))
            _std  = np.load('npy/{}-std.npy'.format(_name))        
            features[i] = (features[i] - _mean) / _std
            
        self.features = np.vstack(features)
        self.labels = np.hstack(labels)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.features[idx]
        feature = feature.reshape(self.feature_shape)

        label = self.labels[idx]
        label = torch.eye(2, dtype=torch.long)[label]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


if __name__ == '__main__':
    benchmark_paths = ['benchmarks/iccad2012_train', 'benchmarks/asml1_train']
    data = MixedLayoutHotspotDataset(benchmark_paths=benchmark_paths)
