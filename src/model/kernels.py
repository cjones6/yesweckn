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
import torch
import torch.nn as nn


class RBFSphere(nn.Module):
    """
    RBF kernel on the sphere.
    """
    def __init__(self, sigma):
        super(RBFSphere, self).__init__()
        self.gamma = 1.0/sigma
        self.name = 'rbf_sphere'

    def forward(self, x, y):
        """
        Evaluate the kernel on the provided inputs
        """
        z = torch.mm(x, y.t())
        gram = torch.exp(-self.gamma**2 * (1-z))
        return gram


class Linear(nn.Module):
    """
    Linear kernel.
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.name = 'linear'

    def forward(self, x, y):
        """
        Evaluate the kernel on the provided inputs.
        """
        return torch.mm(x, y.t())
