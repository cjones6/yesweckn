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
import resource
import torch
import torch.nn as nn

import src.default_params as defaults
import utils

# https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class CKN(nn.Module):
    """
    Convolutional kernel network.
    """
    def __init__(self, layers, input_spatial_dims=None, featurize=None):
        super(CKN, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.featurize = featurize
        self.layers[0].input_spatial_dims = input_spatial_dims

    def _sample_patches(self, data_loader, layer, patches_per_image, patches_per_layer):
        """
        Sample patches from the feature representations of images at the specified layer.
        """
        nimages = 0
        all_patches = []
        for (x, y) in data_loader:
            nimages += len(x)
            x = x.to(defaults.device)
            if self.featurize is not None:
                x = self.featurize(x)
            x = x.type(torch.get_default_dtype())

            if not (layer == 0 and self.layers[0].precomputed_patches):
                for layer_num in range(layer):
                    if layer_num == 0:
                        x = nn.Parameter(x)
                    x = self.layers[layer_num](x)
                patches = utils.extract_some_patches(x, self.layers[layer].patch_size, self.layers[layer].stride,
                                                     patches_per_image, whiten=self.layers[layer].whiten)
            else:
                patches = x.contiguous().view(-1, x.shape[-1])
            patches = patches.data.cpu()
            all_patches.append(patches)

            if nimages > patches_per_layer:
                break

        return torch.cat(all_patches)[:patches_per_layer]

    def _get_filters_dim(self, layer_num, data_loader):
        """
        Get the dimensions of the filters at the specified layer.
        """
        if layer_num == 0:
            x, y = next(iter(data_loader))
            x = x.to(defaults.device)
            if self.featurize is not None:
                x = self.featurize(x)
            if not self.layers[layer_num].precomputed_patches:
                dim1 = x.shape[1]*np.prod(self.layers[layer_num].patch_size)
            else:
                dim1 = x.shape[2]*np.prod(self.layers[layer_num].patch_size)
        else:
            dim1 = self.layers[layer_num - 1].n_filters * self.layers[layer_num].patch_size[0] * self.layers[layer_num].patch_size[1]
        dim2 = self.layers[layer_num].n_filters
        return dim1, dim2

    def init(self, data_loader, patches_per_layer=10000, patches_per_image=1, layers=None):
        """
        Initialize the weights of the CKN at each specified layer. If layers is not specified, it initializes the
        weights at every layer.
        """
        nlayers = len(self.layers)
        if layers is None:
            layers = range(nlayers)
        for layer_num in layers:
            print('Initializing layer', layer_num)
            if self.layers[layer_num].filters_init in ['spherical-k-means']:
                patches = self._sample_patches(data_loader, layer_num, patches_per_image, patches_per_layer)
                self.layers[layer_num].initialize_W(patches)
            else:
                filters_dim = self._get_filters_dim(layer_num, data_loader)
                self.layers[layer_num].initialize_W(torch.zeros(filters_dim))

            if defaults.device.type == 'cuda':
                torch.cuda.empty_cache()

    def forward(self, x):
        """
        Run x through the CKN to generate its feature representation.
        """
        if self.featurize is not None:
            x = self.featurize(x).type(torch.get_default_dtype())
        return self.layers(x)