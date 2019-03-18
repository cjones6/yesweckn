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
import torchvision
from torchvision import transforms

import create_data_loaders


def get_dataloaders(batch_size, valid_size=10000, num_workers=4, transform='std', data_path='../data/MNIST'):
    """
    Create data loaders for MNIST.
    """
    if transform == 'std':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif transform is not None:
        raise NotImplementedError

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    train_loader, valid_loader, train_valid_loader, test_loader = create_data_loaders.generate_dataloaders(
                                                                        train_dataset,
                                                                        test_dataset,
                                                                        valid_dataset=valid_dataset,
                                                                        separate_valid_set=False,
                                                                        valid_size=valid_size,
                                                                        batch_size=batch_size,
                                                                        num_workers=num_workers)

    return train_loader, valid_loader, train_valid_loader, test_loader
