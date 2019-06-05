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
from torch import nn

import src.default_params as defaults
from . import kernels, pool, utils


class CKNLayer(nn.Module):
    """
    Single layer of a CKN.
    """
    def __init__(self, layer_num, patch_size, patch_kernel, n_filters, subsampling_factor, padding=0, stride=(1, 1),
                 precomputed_patches=False, whiten=False, filters_init='spherical-k-means',
                 normalize=True, patch_sigma=0.6, pool_kernel='average', pool_dim=(1, 1),
                 store_normalization=False, kww_reg=0.001, num_newton_iters=20):
        nn.Module.__init__(self)
        self.layer_num = layer_num
        self.patch_size = patch_size
        self.patch_kernel = self._initialize_patch_kernel(patch_kernel, patch_sigma=patch_sigma)
        self.n_filters = n_filters
        self.subsampling_factor = subsampling_factor
        self.pad = padding
        self.stride = stride
        self.precomputed_patches = precomputed_patches
        self.whiten = whiten
        self.filters_init = filters_init
        self.norm = normalize
        self.pooling = pool.Pool(pool_kernel, subsampling_factor, pool_dim)
        self.store_normalization = store_normalization
        self.kww_reg = kww_reg
        self.num_newton_iters = num_newton_iters

        self.W = None
        self.normalization = None
        self.input_spatial_dims = None

    @staticmethod
    def _initialize_patch_kernel(patch_kernel, **kwargs):
        """
        Set the patch kernel for this layer.
        """
        if patch_kernel.lower() == 'rbf_sphere':
            kernel = kernels.RBFSphere(kwargs['patch_sigma'])
        elif patch_kernel.lower() == 'linear':
            kernel = kernels.Linear()
        else:
            raise NotImplementedError

        return kernel

    def initialize_W(self, patches):
        """
        Initialize the filters W.
        """
        if self.filters_init == 'spherical-k-means':
            W = utils.spherical_kmeans(patches, k=self.n_filters)
        elif self.filters_init == 'identity':
            W = torch.eye(patches.shape[1], patches.shape[1], device=defaults.device)
        elif self.filters_init == 'random_normal':
            W = torch.randn(patches.shape[0], patches.shape[1], device=defaults.device)
            W /= torch.norm(W, 2, 0)
            W = W.t()
        elif self.filters_init == 'precomputed':
            W = patches
        else:
            raise NotImplementedError

        self.W = nn.Parameter(W.to(defaults.device), requires_grad=True)

        if self.store_normalization:
            self.normalization = self.compute_normalization().to(defaults.device)

    @staticmethod
    def _normalize_patches(patches, eps=1e-10):
        """
        Normalize the patches so they have norm 1.
        """
        norm_patches = torch.clamp(torch.norm(patches, 2, 1, keepdim=True), min=eps)
        patches = patches / norm_patches
        return patches, norm_patches.squeeze()

    def compute_normalization(self):
        """
        Compute the normalization k(W^TW)^{-1/2}.
        """
        basis_gram = self.patch_kernel(self.W, self.W)
        identity = torch.eye(*basis_gram.shape, device=defaults.device)
        return utils.stable_newton_with_newton(basis_gram + self.kww_reg * identity, maxiter=self.num_newton_iters)

    def _project(self, patches, normalization):
        """
        Perform the projection onto the subspace.
        """
        embedded = self.patch_kernel(patches, self.W)
        embedded = torch.mm(embedded, normalization)
        return embedded

    def _next_dims(self, f):
        """
        Get the dimensions of the feature representation after applying this layer.
        """
        batch_size = f.shape[0]
        if self.precomputed_patches is False:
            height, width = f.shape[2:4]
        else:
            height = self.input_spatial_dims[0]
            width = self.input_spatial_dims[1]
        height_next = int((height + 2 * self.pad - (self.patch_size[0] - 1) - 1) / self.stride[0] + 1)
        width_next = int((width + 2 * self.pad - (self.patch_size[1] - 1) - 1) / self.stride[1] + 1)
        return batch_size, height_next, width_next

    def forward(self, f):
        """
        Apply this layer of the CKN to the provided input features.
        """
        if self.W is None:
            raise AssertionError('You must initialize W prior to running forward()')

        if self.precomputed_patches is False:
            patches_by_image = utils.images_to_patches(f, self.patch_size, self.stride, whiten=self.whiten,
                                                       padding=self.pad)
        else:
            patches_by_image = f
        patches = patches_by_image.contiguous().view(-1, patches_by_image.shape[-1])

        if self.norm:
            patches, norm_patches = self._normalize_patches(patches)
        if self.normalization is None or self.store_normalization is False:
            normalization = self.compute_normalization().to(defaults.device)
        else:
            normalization = self.normalization.to(defaults.device)

        cutoff = 113246208  # be careful with gpu memory
        if patches.shape[0]*self.W.shape[1] < cutoff:
            projected_patches = self._project(patches, normalization)
        else:
            bsize = int(cutoff/patches.shape[1])
            idx_num = 0
            projected_patches = []
            while idx_num < patches.shape[0]:
                subpatches = patches[idx_num:idx_num+bsize]
                projected_patches.append(self._project(subpatches, normalization))
                idx_num += bsize
            projected_patches = torch.cat(projected_patches)

        if self.norm:
            if norm_patches.ndimension() > 0:
                projected_patches = projected_patches*norm_patches.unsqueeze(1)
            else:
                projected_patches = projected_patches*norm_patches

        all_patches = utils.patches_to_images(projected_patches, self._next_dims(f))

        f_next = self.pooling(all_patches)

        return f_next
