import torch
import torch.nn as nn


class LeNet1(nn.Module):
    """
    Reimplementation of the original LeNet-1 (excluding the output layer) with nfilters filters per layer.

    References:
        - Y. LeCun, B. E. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. E. Hubbard, and L. D. Jackel. Handwritten
          digit recognition with a back-propagation network. In Advances in Neural Information Processing Systems, pages
          396–404, 1989
        - Y. LeCun, L. Jackel, L. Bottou, A. Brunot, C. Cortes, J. Denker, H. Drucker, I. Guyon, U. Müller,
          E. Säckinger, P. Simard, and V. Vapnik. Comparison of learning algorithms for handwritten digit recognition.
          In International Conference on Artificial Neural Networks, pages 53–60, 1995.
        - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.
          In Intelligent Signal Processing, pages 306–351. IEEE Press, 2001.
    """
    def __init__(self, nfilters, locally_connected_layer=True, bias=True):
        super(LeNet1, self).__init__()
        if nfilters is not None and locally_connected_layer:
            raise NotImplementedError
        if nfilters is None:
            self.filter_sizes = [4, 12]
        else:
            self.filter_sizes = [nfilters]*2

        sigmoid = ScaledTanh()

        H1 = torch.nn.Conv2d(1, self.filter_sizes[0], kernel_size=5, stride=1, padding=0, bias=bias)
        H2 = TrainablePool(self.filter_sizes[0], size=2, stride=2, padding=0, bias=bias)
        if locally_connected_layer:
            H3 = LeNetC3Layer('lenet-1', bias=bias)
        else:
            H3 = torch.nn.Conv2d(self.filter_sizes[0], self.filter_sizes[1], kernel_size=5, stride=1, padding=0,
                                 bias=bias)
        H4 = TrainablePool(self.filter_sizes[1], size=2, stride=2, padding=0, bias=bias)

        self.layers = nn.Sequential(H1, sigmoid, H2, sigmoid, H3, sigmoid, H4, sigmoid)

    def forward(self, x):
        return self.layers(x)


class LeNet5(nn.Module):
    """
    Reimplementation of the original LeNet-5 (excluding the output layer) with nfilters filters per layer.

    Reference:
        - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.
          In Intelligent Signal Processing, pages 306–351. IEEE Press, 2001.
    """
    def __init__(self, nfilters=None, locally_connected_layer=True, bias=True):
        super(LeNet5, self).__init__()
        if nfilters is not None and locally_connected_layer:
            raise NotImplementedError
        if nfilters is None:
            self.filter_sizes = [6, 16, 120, 84]
        else:
            self.filter_sizes = [nfilters]*4

        sigmoid = ScaledTanh()

        C1 = torch.nn.Conv2d(1, self.filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=bias)
        S2 = TrainablePool(self.filter_sizes[0], size=2, stride=2, padding=0, bias=bias)
        if locally_connected_layer:
            C3 = LeNetC3Layer('lenet-5', bias=bias)
        else:
            C3 = torch.nn.Conv2d(self.filter_sizes[0], self.filter_sizes[1], kernel_size=5, stride=1, padding=0,
                                 bias=bias)
        S4 = TrainablePool(self.filter_sizes[1], size=2, stride=2, padding=0, bias=bias)
        C5 = torch.nn.Conv2d(self.filter_sizes[1], self.filter_sizes[2], kernel_size=5, stride=1, padding=0, bias=bias)
        reshape = Reshape()
        F6 = torch.nn.Linear(self.filter_sizes[2], self.filter_sizes[3], bias=bias)

        self.layers = nn.Sequential(C1, sigmoid, S2, sigmoid, C3, sigmoid, S4, sigmoid, C5, sigmoid, reshape, F6,
                                    sigmoid)

    def forward(self, x):
        return self.layers(x)


class ScaledTanh(nn.Module):
    """
    Scaled Tanh nonlinearity as described in LeCun et al. (2001).
    """
    def __init__(self, A=1.7159, S=2/3):
        nn.Module.__init__(self)
        self.A = A
        self.S = S

    def forward(self, x):
        return self.A*torch.tanh(self.S*x)


class TrainablePool(nn.Module):
    """
    Pooling followed by multiplication by one weight per feature map and then the addition of one bias per feature map.
    """
    def __init__(self, in_channels, size, stride, padding, bias=True):
        nn.Module.__init__(self)
        self.use_bias = bias
        self.pool = torch.nn.AvgPool2d(size, stride=stride, padding=padding, divisor_override=1)
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_channels, 1, 1))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels, 1, 1))

    def forward(self, x):
        if self.use_bias:
            return self.weight*self.pool(x)+self.bias
        else:
            return self.weight*self.pool(x)


class Reshape(nn.Module):
    """
    Reshape the input to 2D by vectorizing all but the first dimension.
    """
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LeNetC3Layer(nn.Module):
    """
    Locally connected C3 layers for LeNet-1 and LeNet-5.
    """
    def __init__(self, lenet_name, bias=True):
        super(LeNetC3Layer, self).__init__()
        if lenet_name.lower() == 'lenet-1':
            self.connections = self.lenet1_c3_connections()
        elif lenet_name.lower() == 'lenet-5':
            self.connections = self.lenet5_c3_connections()

        convs = []
        for i in range(self.connections.shape[1]):
            nonzero_idxs = torch.nonzero(self.connections[:, i])
            conv = torch.nn.Conv2d(len(nonzero_idxs), 1, kernel_size=5, stride=1, padding=0, bias=bias)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    @staticmethod
    def lenet1_c3_connections():
        connections = torch.zeros((4, 12))
        idxs = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8),
                (2, 10), (2, 11), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)]
        for idx in idxs:
            connections[idx] = 1
        return connections

    @staticmethod
    def lenet5_c3_connections():
        connections = torch.zeros((6, 16))
        idxs = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (5, 3), (0, 4),
                (4, 4), (5, 4), (0, 5), (1, 5), (5, 5), (0, 6), (1, 6), (2, 6), (3, 6), (1, 7), (2, 7), (3, 7), (4, 7),
                (2, 8), (3, 8), (4, 8), (5, 8), (0, 9), (3, 9), (4, 9), (5, 9), (0, 10), (1, 10), (4, 10), (5, 10),
                (0, 11), (1, 11), (2, 11), (5, 11), (0, 12), (1, 12), (3, 12), (4, 12), (1, 13), (2, 13), (4, 13),
                (5, 13), (0, 14), (2, 14), (3, 14), (5, 14), (0, 15), (1, 15), (2, 15), (3, 15), (4, 15), (5, 15)]
        for idx in idxs:
            connections[idx] = 1
        return connections

    def forward(self, x):
        return torch.cat([self.convs[i](x[:, torch.nonzero(self.connections[:, i]), :, :].squeeze(2))
                          for i in range(len(self.convs))], dim=1)


def init_uniform(m, low=-2.4, high=2.4):
    """
    Initialize the weights of a layer m with draws from the uniform distribution between low/fan_in and high/fan_in
    (fan_in=number of inputs).
    """
    try:
        if m.weight.dim() > 1:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        else:
            fan_in = 1
        m.weight.data.uniform_(low/fan_in, high/fan_in)
        m.bias = nn.Parameter(torch.zeros_like(m.bias))
    except:
        pass


def init_normal(m, std=0.2):
    """
    Initialize the weights of a layer m with draws from the random normal distribution with mean 0 and the given
    standard deviation.
    """
    try:
        m.weight.data.normal_(0.0, std)
        m.bias = nn.Parameter(torch.zeros_like(m.bias))
    except:
        pass
