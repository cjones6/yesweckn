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
import numpy as np
import torch
import torch.nn as nn

import src.default_params as defaults


class Pool(nn.Module):
    """
    Pooling for the CKN.
    """
    def __init__(self, kernel, subsample_factor=1, pool_dim=(1, 1)):
        super(Pool, self).__init__()
        self.kernel = kernel
        self.subsample_factor = subsample_factor
        self.pool_dim = pool_dim
        self.weights = self._get_weights()

    def _get_weights(self):
        """
        Compute the weights for the pooling.
        """
        if self.kernel == 'average':
            weights = 1/np.prod(self.pool_dim)*np.ones(self.pool_dim)
        else:
            raise NotImplementedError

        weights = torch.Tensor(weights[np.newaxis, np.newaxis, :, :]).to(defaults.device)
        return weights

    def forward(self, x):
        """
        Perform pooling on the input x.
        """
        batch_size = x.shape[0]
        if self.weights.nelement() != 1:
            inner_approx = x.contiguous().view(-1, x.shape[2], x.shape[3]).unsqueeze(1)
            outer_approx = nn.functional.conv2d(inner_approx, self.weights, stride=1, padding=0)
            outer_approx = outer_approx.contiguous().view(batch_size, -1, outer_approx.shape[2], outer_approx.shape[3])
        else:
            outer_approx = x
        offset = int(np.ceil(self.subsample_factor / 2)) - 1
        return outer_approx[:, :, offset::self.subsample_factor, offset::self.subsample_factor]
