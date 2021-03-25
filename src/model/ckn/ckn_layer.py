import torch
from torch import nn

import src.default_params as defaults
from src.model.ckn import utils, kernels, pool


class CKNLayer(nn.Module):
    """
    Single layer of a CKN.
    """
    def __init__(self, layer_num, patch_size, patch_kernel, n_filters, subsampling_factor, padding=0, stride=(1, 1),
                 precomputed_patches=False, whiten=False, filters_init='spherical-k-means', normalize=True,
                 patch_sigma=0.6, matern_order=1.5, pool_kernel='average', pool_dim=(1, 1), store_normalization=False,
                 kww_reg=0.001, num_newton_iters=20):
        nn.Module.__init__(self)
        self.layer_num = layer_num
        self.patch_size = patch_size
        self.patch_kernel = self._initialize_patch_kernel(patch_kernel, patch_sigma=patch_sigma,
                                                          matern_order=matern_order)
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
        elif patch_kernel.lower() == 'rbf':
            kernel = kernels.RBF(kwargs['patch_sigma'])
        elif patch_kernel.lower() == 'linear':
            kernel = kernels.Linear()
        elif patch_kernel.lower() == 'matern_sphere':
            kernel = kernels.HalfMatern(kwargs['matern_order'], kwargs['patch_sigma'])
        else:
            raise NotImplementedError

        return kernel

    def initialize_W(self, patches):
        """
        Initialize the filters W.
        """
        if self.filters_init == 'spherical-k-means':
            W = utils.spherical_kmeans(patches, k=self.n_filters)
        elif self.filters_init == 'k-means':
            W = utils.kmeans(patches, k=self.n_filters)
        elif self.filters_init == 'random_sample':
            W = utils.random_sample(patches, k=self.n_filters)
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

    def compute_normalization(self, eps=1e-12):
        """
        Compute the normalization (k(WW^T)+kww_reg I)^{-1/2}.
        """
        basis_gram = self.patch_kernel(self.W, self.W)

        if self.num_newton_iters > 0:
            identity = torch.eye(*basis_gram.shape, device=basis_gram.device)
            return utils.stable_newton_with_newton(basis_gram + self.kww_reg * identity, maxiter=self.num_newton_iters)
        else:
            U, S, V = torch.svd(basis_gram)
            sqrt_S = torch.sqrt(torch.clamp(S, min=eps) + self.kww_reg)
            return torch.mm(torch.div(U, sqrt_S), V.t())

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
            normalization = self.compute_normalization()
        else:
            normalization = self.normalization

        projected_patches = self._project(patches, normalization)

        if self.norm:
            if norm_patches.ndimension() > 0:
                projected_patches = projected_patches*norm_patches.unsqueeze(1)
            else:
                projected_patches = projected_patches*norm_patches

        all_patches = utils.patches_to_images(projected_patches, self._next_dims(f))
        f_next = self.pooling(all_patches)

        return f_next
