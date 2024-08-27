import os
import numpy as np

from data import data as data_dct

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LayoutHotspotDataset(Dataset):
    '''EDA layout hotspot dataset.'''

    def __init__(self, features, labels, transform='auto', normalize=None, resample=True, sel_ch=[]):
        self.feature_buffer = features.astype(np.float32)
        self.label_buffer = labels.astype(np.long)

        if sel_ch:
            self.feature_buffer = self.feature_buffer[...,sel_ch]

        mean = self.feature_buffer.mean(axis=(0, 1))
        std = self.feature_buffer.std(axis=(0, 1))

        if normalize is not None:
            self.mean, self.std = normalize
            if sel_ch:
                self.mean = self.mean[sel_ch]
                self.std = self.std[sel_ch]

        # convert to one-hot format
        # self.label_buffer = np.eye(2)[self.label_buffer.astype(np.long)]

        assert self.feature_buffer.shape[0] == self.label_buffer.shape[0]

        print("Allocated dataset with size", self.feature_buffer.shape)

        if resample and transform != 'test':
            pos_idx = self.label_buffer == 1
            neg_idx = self.label_buffer == 0
            if pos_idx.sum() < neg_idx.sum():
                resampled_ft = np.stack(self.feature_buffer[pos_idx])
                resampled_lb = np.stack(self.label_buffer[pos_idx])
                # print(resampled_lb.shape)
                n_repeat = neg_idx.sum() // pos_idx.sum() - 1
                resampled_ft = np.tile(resampled_ft, (n_repeat, 1, 1))
                resampled_lb = np.tile(resampled_lb, (n_repeat,))
                # print(resampled_lb.shape)
                # print(self.label_buffer.shape)
                self.feature_buffer = np.vstack((self.feature_buffer, resampled_ft))
                self.label_buffer   = np.hstack((self.label_buffer, resampled_lb))
            else:
                resampled_ft = np.stack(self.feature_buffer[neg_idx])
                resampled_lb = np.stack(self.label_buffer[neg_idx])
                n_repeat = pos_idx.sum() // neg_idx.sum() - 1
                resampled_ft = np.tile(resampled_ft, (n_repeat, 1, 1))
                resampled_lb = np.tile(resampled_lb, (n_repeat))
                self.feature_buffer = np.vstack((self.feature_buffer, resampled_ft))
                self.label_buffer   = np.hstack((self.label_buffer, resampled_lb))
            print("Resampled dataset to size", self.feature_buffer.shape)

        self.feature_shape = (12, 12, -1)   # '-1' should be 32 channels

        print("Using transform option:", transform)
        if transform == 'auto':
            # use default local transform
            # compute mean & std for normalization
            _mean = self.feature_buffer.mean(axis=(0, 1))
            _std  = self.feature_buffer.std(axis=(0, 1))
            assert _mean.shape == (32,), 'Check feature buffer!'
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_mean, _std)
            ])
        elif transform == 'train':
            assert self.mean is not None and self.std is not None
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        elif transform == 'test':
            assert self.mean is not None and self.std is not None
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            # use passed in transform
            self.transform = transform

        n_pos = sum(self.label_buffer)
        print('#pos = {}, #neg = {}'.format(n_pos, len(self.label_buffer) - n_pos))


    def __len__(self):
        return self.feature_buffer.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.feature_buffer[idx]
        feature = feature.reshape(self.feature_shape)

        label = self.label_buffer[idx]
        label = torch.eye(2, dtype=torch.long)[label]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


def alloc_benchmark(benchmark_dir, clients_num, transform='train', normalize=None, sel_ch=[]):
    '''
    Splits a benchmark into training sets of clients
    :arg benchmark_dir: dir to the benchmark to be splitted
    :arg clients_num: number of clients after splitting
    :return clients_train_set_list: a list of LayoutHotspotDataset
    '''
    if clients_num == 0:
        return []

    # sel_ch = [26, 17, 24, 23, 6, 20, 12, 18, 16, 31, 30,  7, 19,  4,  5,  9, 28, 29,  8,  3, 11, 14, 15,  1,
    #           10, 27,  2, 21, 0]      # L2 norm only
    # sel_ch = [1, 15, 23,  7, 18, 26, 13,  9, 31, 11,  5, 20, 16, 25, 24, 19, 22, 30, 28,  8, 21,  2, 17, 29,
    #           3, 27,  4, 12,  0] # group lasso 29
    # sel_ch = [7, 18, 26, 13,  9, 31, 11,  5, 20, 16, 25, 24, 19, 22, 30, 28,  8, 21,  2, 17, 29,
    #           3, 27,  4, 12,  0] # group lasso 26
    # sel_ch = [13,  9, 31, 11,  5, 20, 16, 25, 24, 19, 22, 30, 28,  8, 21,  2, 17, 29,
    #           3, 27,  4, 12,  0]  # group lasso 23
    # sel_ch = [11,  5, 20, 16, 25, 24, 19, 22, 30, 28,  8, 21,  2, 17, 29,
    #           3, 27,  4, 12,  0]  # group lasso 20

    label_file = os.path.join(benchmark_dir, 'label.csv')
    data = data_dct(benchmark_dir, label_file, preload=True)

    # lists of features & labels
    features = np.stack(data.ft_buffer)
    labels   = np.stack(data.label_buffer)
    del data

    '''
    Make even splits
    '''
    idx = np.arange(features.shape[0])
    np.random.shuffle(idx)

    n_samples_per_split = features.shape[0] // clients_num

    base = 0
    data_list = []
    for i in range(clients_num - 1):
        _idx = idx[base:base+n_samples_per_split]
        data_list.append(LayoutHotspotDataset(
            features=features[_idx],
            labels=labels[_idx],
            transform=transform,
            normalize=normalize,
            sel_ch=sel_ch
        ))
        base += n_samples_per_split
    # last split
    data_list.append(LayoutHotspotDataset(
        features=features[idx[base:]],
        labels=labels[idx[base:]],
        transform=transform,
        normalize=normalize,
        sel_ch=sel_ch
    ))

    return data_list
