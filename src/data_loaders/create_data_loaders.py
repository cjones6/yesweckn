# Copyright (c) 2019 Corinne Jones, Vincent Roulet, Zaid Harchaoui.
#
# This file is part of yesweckn. yesweckn provides an implementation
# of the CKNs used in the following paper:
#
# C. Jones, V. Roulet and Z. Harchaoui. Kernel-based Translations
# of Convolutional Networks. In arXiv, 2019.
#
# yesweckn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# yesweckn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with yesweckn.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import division
import torch.utils.data
import numpy as np
import random
import torch

import src.default_params as defaults


def generate_dataloaders(train_dataset, test_dataset, valid_dataset=None, separate_valid_set=False, valid_size=0,
                         batch_size=128, num_workers=0):
    """
    Create data loaders given the corresponding datasets from the PyTorch Dataset class.
    """
    train_sampler = StratifiedSampler(train_dataset, batch_size=batch_size)

    if separate_valid_set:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False)
        train_stratified_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                              sampler=train_sampler, num_workers=num_workers,
                                                              pin_memory=True, drop_last=False)
    elif valid_size > 0:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        train_novalid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[valid_size:])
        valid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[:valid_size])

        train_loader = torch.utils.data.DataLoader(train_novalid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False)
        train_valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                                         num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False)
        train_stratified_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                              sampler=train_sampler, num_workers=num_workers,
                                                              pin_memory=True, drop_last=False)
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False, pin_memory=True)

    if not separate_valid_set and valid_size > 0:
        return train_loader, valid_loader, train_valid_loader, test_loader
    else:
        return train_loader, valid_loader, train_stratified_loader, test_loader


class StratifiedSampler(torch.utils.data.Sampler):
    """
    Sampler to create batches with balanced classes. Warning: This assumes each class is equally represented.
    """
    def __init__(self, dataset, batch_size):
        self.labels = self._get_labels(dataset)
        self.n_splits = int(len(self.labels) / batch_size)
        self.batch_size = batch_size
        self.nclasses = len(torch.unique(self.labels))
        self.im_per_class = len(self.labels)/self.nclasses
        self.class_idxs = [np.where(self.labels == i)[0] for i in range(self.nclasses)]

    @staticmethod
    def _get_labels(dataset):
        if 'tensors' in dataset.__dict__.keys():
            labels = dataset.tensors[1]
        elif 'train_labels' in dataset.__dict__.keys():
            labels = dataset.train_labels
        elif 'targets' in dataset.__dict__.keys():
            labels = dataset.targets
        elif 'labels' in dataset.__dict__.keys():
            labels = dataset.labels
        elif 'dataset' in dataset.__dict__.keys():
            labels = dataset.dataset.train_labels[dataset.indices]
        else:
            raise NotImplementedError

        if isinstance(labels, list):
            labels = torch.IntTensor(labels)
        return labels

    def _sample_idxs(self):
        idxs = []
        for i in range(self.nclasses):
            cls_idxs = np.random.choice(self.class_idxs[i], int(np.ceil(self.batch_size/self.nclasses)), replace=False)
            idxs.extend(cls_idxs)

        np.random.shuffle(idxs)
        return idxs[:self.batch_size]

    def __iter__(self):
        idxs = np.concatenate([self._sample_idxs() for _ in range(self.n_splits)]).tolist()
        return iter(idxs)

    def __len__(self):
        return len(self.labels)


class PreloadedDataset(torch.utils.data.Dataset):
    """
    Class for data that has already been loaded into memory that allows for transformations.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
