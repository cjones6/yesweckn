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
from __future__ import print_function
import glob
import numpy as np
import os
import sys
import torch
import torchvision
from torchvision import transforms

from . import create_data_loaders


def get_dataloaders(batch_size, valid_size=10000, num_workers=4, transform='std', data_path='../data/cifar10',
                    precomputed_patches=False):
    """
    Create data loaders for CIFAR-10.
    """
    if transform == 'std':
        mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    elif transform is not None:
        raise NotImplementedError

    if precomputed_patches is False:
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    else:
        print('Loading pre-computed patches into memory... ', end='')
        python_version = sys.version.split(' ')[0]
        dataset = {'train_patches': None, 'train_labels': None, 'test_patches': None, 'test_labels': None}
        for dataset_name in dataset.keys():
            dataset_files = sorted(glob.glob1(data_path, dataset_name + '*' + python_version + '*'))
            data = []
            for filename in dataset_files:
                subset = np.load(os.path.join(data_path, filename))
                if 'patches' in filename:
                    subset = subset.float()
                data.append(subset)
            dataset[dataset_name] = torch.cat(data)
        print('done')

        train_dataset = create_data_loaders.PreloadedDataset(dataset['train_patches'], dataset['train_labels'])
        valid_dataset = create_data_loaders.PreloadedDataset(dataset['train_patches'], dataset['train_labels'])
        test_dataset = create_data_loaders.PreloadedDataset(dataset['test_patches'], dataset['test_labels'])

    train_loader, valid_loader, train_valid_loader, test_loader = create_data_loaders.generate_dataloaders(
                                                                        train_dataset,
                                                                        test_dataset,
                                                                        valid_dataset=valid_dataset,
                                                                        separate_valid_set=False,
                                                                        valid_size=valid_size,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers)

    return train_loader, valid_loader, train_valid_loader, test_loader
