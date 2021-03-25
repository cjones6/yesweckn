import glob
import numpy as np
import os
import sys
import torch
import torchvision
from torchvision import transforms

from . import create_data_loaders


def get_dataloaders(valid_size=10000, transform='std', batch_size=128, data_path='../data/cifar10', num_workers=0,
                    precomputed_patches=True):
    """
    Create data loaders for whitened patches from CIFAR-10.
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
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

        try:
            train_dataset = create_data_loaders.PreloadedDataset(train_dataset.data, np.array(train_dataset.targets),
                                                                 transform=transform)
            test_dataset = create_data_loaders.PreloadedDataset(test_dataset.data, np.array(test_dataset.targets),
                                                                transform=transform)
        except:
            train_dataset = create_data_loaders.PreloadedDataset(train_dataset.train_data,
                                                                 np.array(train_dataset.train_labels),
                                                                 transform=transform)
            test_dataset = create_data_loaders.PreloadedDataset(test_dataset.test_data,
                                                                np.array(test_dataset.test_labels),
                                                                transform=transform)
        return create_data_loaders.generate_dataloaders(train_dataset,
                                                        test_dataset,
                                                        separate_valid_set=False,
                                                        valid_size=valid_size,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers)
    else:
        print('Loading pre-computed patches into memory... ', end='')
        python_version = sys.version.split(' ')[0]
        dataset = {'train_patches': None, 'train_labels': None, 'test_patches': None, 'test_labels': None}
        for dataset_name in dataset.keys():
            dataset_files = sorted(glob.glob1(data_path, dataset_name + '*' + python_version + '*'))
            if len(dataset_files) == 0:
                raise Exception('Precomputed patches not found. Please make sure you ran the script ' \
                                 'misc/prewhiten_cifar10.py.')
            data = []
            for filename in dataset_files:
                subset = np.load(os.path.join(data_path, filename), allow_pickle=True)
                if 'patches' in filename:
                    subset = subset.float()
                data.append(subset)
            dataset[dataset_name] = torch.cat(data)
        print('done')

        train_dataset = create_data_loaders.PreloadedDataset(dataset['train_patches'], dataset['train_labels'].numpy())
        test_dataset = create_data_loaders.PreloadedDataset(dataset['test_patches'], dataset['test_labels'].numpy())

        return create_data_loaders.generate_dataloaders(train_dataset, test_dataset, separate_valid_set=False,
                                                        valid_size=valid_size, batch_size=batch_size,
                                                        num_workers=num_workers)
