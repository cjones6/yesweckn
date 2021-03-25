import numpy as np
import torch
import torch.nn as nn

import src.default_params as defaults


class Pool(nn.Module):
    """
    Pooling for the CKN.
    """
    def __init__(self, kernel, subsample_factor=1, pool_dim=(1, 1), pool_sigma=None):
        super(Pool, self).__init__()
        self.kernel = kernel
        self.subsample_factor = subsample_factor
        self.pool_dim = pool_dim
        self.pool_sigma = pool_sigma
        self.weights = self._get_weights()

    def _get_weights(self):
        """
        Compute the weights for the pooling.
        """
        if self.kernel == 'average':
            weights = 1/np.prod(self.pool_dim)*np.ones(self.pool_dim)
        elif self.kernel == 'RBF':
            if self.pool_dim[0] is None:
                self.pool_dim[0] = self.subsample_factor
            if self.pool_sigma is None:
                self.pool_sigma = self.subsample_factor
            if self.pool_dim[0] != 0:
                weights = np.array([np.exp(-x**2/(self.pool_sigma**2)) for x in range(-int(np.ceil(self.pool_dim[0])),
                                                                             int(np.ceil(self.pool_dim[0])) + 1)])
                weights = np.outer(weights/sum(weights), weights/sum(weights))
            else:
                weights = np.array([[1]])
        elif self.kernel == 'max':
            weights = None
        else:
            raise NotImplementedError

        if weights is not None:
            weights = torch.Tensor(weights[np.newaxis, np.newaxis, :, :])
        return weights

    def forward(self, x):
        """
        Perform pooling on the input x.
        """
        batch_size = x.shape[0]
        if self.weights is None and self.kernel == 'max':
            outer_approx = nn.functional.max_pool2d(x, self.pool_dim, stride=1, padding=0)
        elif self.weights is None:
            raise NotImplementedError
        elif self.weights.nelement() != 1:
            self.weights = self.weights.to(x.device)
            if self.kernel == 'RBF':
                padding = int(self.weights.shape[2] / 2)
            else:
                padding = 0

            inner_approx = x.contiguous().view(-1, x.shape[2], x.shape[3]).unsqueeze(1)
            outer_approx = nn.functional.conv2d(inner_approx, self.weights, stride=1, padding=padding)
            outer_approx = outer_approx.contiguous().view(batch_size, -1, outer_approx.shape[2], outer_approx.shape[3])
        else:
            outer_approx = x
        offset = int(np.ceil(self.subsample_factor / 2)) - 1
        return outer_approx[:, :, offset::self.subsample_factor, offset::self.subsample_factor]
