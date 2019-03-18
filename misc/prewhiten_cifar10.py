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
import numpy as np
import os
import cPickle as pickle
import sys
import torch

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append('../src')
sys.path.append('..')

import src.data_loaders.cifar10
import src.model.utils as utils

python_version = sys.version.split(' ')[0]
cifar_path = '../data/cifar10'
outdir = '../data/cifar10_whitened'
batch_size = 50
save_every = 100

if not os.path.exists(outdir):
    os.makedirs(outdir)

# Create data loaders
train_loader, valid_loader, _, test_loader = src.data_loaders.cifar10.get_dataloaders(batch_size=batch_size, precomputed_patches=False,
                                                                                      data_path=cifar_path)

train_valid_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(
                                                    [train_loader.dataset, valid_loader.dataset]
                                                 ),
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

# Generate the whitened patches and save them with their labels
for loadernum, loader in enumerate([train_valid_loader, test_loader]):
    all_patches = []
    all_ys = []
    nimages = 0
    if loadernum == 0:
        print('Extracting and whitening patches from the training set')
    else:
        print('\nExtracting and whitening patches from the test set')
    total_images = len(loader.dataset)
    for i, (x, y) in enumerate(loader):
        x = x.double().cuda()
        x = utils.pad(x, 1)
        patches_by_image = utils.images_to_patches(x, np.array([3, 3]), np.array([1, 1]), whiten=True)
        all_patches.append(patches_by_image.cpu().detach())
        all_ys.append(y)
        nimages += len(y)
        print('\r%0.2f' % (nimages/total_images*100), '% done', end='')
        if loadernum == 0 and i > 0 and i % save_every == 0:
            all_patches = torch.cat(all_patches).float()
            all_ys = torch.cat(all_ys)
            pickle.dump(all_patches, open(os.path.join(outdir, 'train_patches' + str(i // save_every) + '_python' + python_version + '.pickle'), 'wb'), protocol=-1)
            pickle.dump(all_ys, open(os.path.join(outdir, 'train_labels' + str(i // save_every) + '_python' + python_version + '.pickle'), 'wb'), protocol=-1)
            all_patches = []
            all_ys = []
    if loadernum == 0:
        all_patches = torch.cat(all_patches)
        all_ys = torch.cat(all_ys)
        pickle.dump(all_patches, open(os.path.join(outdir, 'train_patches' + str(1 + i // save_every) + '_python' + python_version + '.pickle'), 'wb'), protocol=-1)
        pickle.dump(all_ys, open(os.path.join(outdir, 'train_labels' + str(1 + i // save_every) + '_python' + python_version + '.pickle'), 'wb'), protocol=-1)
    else:
        all_patches = torch.cat(all_patches).float()
        all_ys = torch.cat(all_ys)
        pickle.dump(all_patches, open(os.path.join(outdir, 'test_patches_python' + python_version + '.pickle'), 'wb'), protocol=-1)
        pickle.dump(all_ys, open(os.path.join(outdir, 'test_labels_python' + python_version + '.pickle'), 'wb'), protocol=-1)
