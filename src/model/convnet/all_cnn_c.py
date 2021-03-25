import torch
import torch.nn as nn

from src import default_params as defaults


class AllCNNC(nn.Module):
    """
    Reimplementation of the original All-CNN-C without dropout and with nfilters filters per layer. The input is
    assumed to be whitened 3x3x3 patches.

    Reference:
        - J.T. Springenberg, A. Dosovitskiy, T. Brox, M.A. Riedmiller. Striving for Simplicity: The All Convolutional
          Net. In Proceedings of the International Conference on Learning Representations (Workshop track), 2015
    """
    def __init__(self, nfilters=None, bias=True):
        super(AllCNNC, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(27, nfilters, bias=bias),
            Reshape((-1, 32, 32, nfilters)),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=2, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=2, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nfilters, nfilters, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.shape[0], x.shape[1], -1)
        x = torch.mean(x, 2)
        return x


class Reshape(nn.Module):
    """
    Reshape the input to 4D with the given input shape.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape).permute(0, 3, 1, 2)


def init_weights(m):
    """
    Initialize the weights of a layer m.
    """
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.normal_(0, 0.3)
        m.bias = nn.Parameter(torch.zeros_like(m.bias).to(defaults.device))


def all_cnn_c(**kwargs):
    """
    Load the All-CNN-C architecture and initialize the weights.
    """
    model = AllCNNC(**kwargs)
    model.apply(init_weights)
    return model
