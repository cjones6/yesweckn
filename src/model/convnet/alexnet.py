import torch
import torch.nn as nn

from src import default_params as defaults


class AlexNet(nn.Module):
    """
    Reimplementation of the AlexNet without local response normalization and with nfilters filters per layer.

    Reference:
        - A. Krizhevsky. One weird trick for parallelizing convolutional neural networks. CoRR abs/1404.5997, 2014.
    """
    def __init__(self, nfilters, bias=True):
        super(AlexNet, self).__init__()
        self.layers = nn.Sequential(
        nn.Conv2d(3, nfilters, kernel_size=11, stride=4, padding=2, bias=bias),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(nfilters, nfilters, kernel_size=5, padding=2, bias=bias),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(nfilters, nfilters, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(inplace=True),
        nn.Conv2d(nfilters, nfilters, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(inplace=True),
        nn.Conv2d(nfilters, nfilters, kernel_size=3, padding=1, bias=bias),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        Reshape(),
        nn.Linear(nfilters * 6 * 6, nfilters),
        nn.ReLU(inplace=True),
        nn.Linear(nfilters, nfilters),
        nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def init_weights(m):
    """
    Initialize the weights of a layer m.
    """
    try:
        m.weight.data.normal_(0.0, .1).cuda()
        m.bias = nn.Parameter(torch.zeros_like(m.bias).to(defaults.device))
    except:
        pass


class Reshape(nn.Module):
    """
    Reshape the input to 2D by vectorizing all but the first dimension.
    """
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def alexnet(**kwargs):
    """
    Load the AlexNet architecture and initialize the weights.
    """
    model = AlexNet(**kwargs)
    model.apply(init_weights)
    return model
