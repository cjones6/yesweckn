import torch
import torch.nn as nn

from src import default_params as defaults


class RBF(nn.Module):
    """
    RBF kernel.
    """
    def __init__(self, sigma):
        super(RBF, self).__init__()
        self.sigma = sigma
        self.name = 'rbf'

    def forward(self, x, y, **kwargs):
        norm = squared_l2_norm(x, y)
        gram = torch.exp(-norm*1/(2*self.sigma**2))

        return gram


class RBFSphere(nn.Module):
    """
    RBF kernel on the sphere.
    """
    def __init__(self, sigma):
        super(RBFSphere, self).__init__()
        self.gamma = 1.0/sigma
        self.name = 'rbf_sphere'

    def forward(self, x, y):
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
        return torch.mm(x, y.t())


class HalfMaternFunction(torch.autograd.Function):
    """
    Matern kernel function with custom backward. The value nu is assumed to be 0.5+n where n is an integer.
    """
    @staticmethod
    def forward(ctx, input, nu, sigma):
        ctx.constant = [nu, sigma]

        input2 = 2 * torch.sqrt(nu * torch.clamp(1 - input, min=0)) / sigma
        ctx.save_for_backward(input2)

        return normalized_modif_bessel(input2, nu)

    @staticmethod
    def backward(ctx, grad_output):
        input2, = ctx.saved_tensors
        [nu, sigma] = ctx.constant

        grad_input = (nu / ((nu-1)*sigma**2) * normalized_modif_bessel(input2, nu - 1)) * grad_output
        grad_nu = None
        grad_sigma = None
        return grad_input, grad_nu, grad_sigma


class HalfMatern(nn.Module):
    """
    Matern kernel. The value nu is assumed to be 0.5+n where n is an integer.
    """
    def __init__(self, nu, sigma):
        super(HalfMatern, self).__init__()
        self.name = 'matern_sphere'
        self.nu = nu
        self.sigma = sigma

    def forward(self, x, y):
        z = torch.mm(x, y.t())
        return HalfMaternFunction.apply(z, self.nu, self.sigma)


def normalized_modif_bessel(zz, nu):
    """
    Normalized modified Bessel function of the second kind of order nu.
    """
    if (nu-0.5) % 1 != 0:
        raise NotImplementedError
    p = int(nu-0.5)
    out = 0
    for j in range(p+1):
        out = out + factorial_frac(p-j, p)/(factorial(j)*factorial_frac(2*p-j, 2*p))*(2*zz)**j
    out = out*torch.exp(-zz)
    return out


def squared_l2_norm(x, y):
    """
    Compute ||x-y||^2 for every pair of rows in x and y.
    """
    nx = x.size(0)
    ny = y.size(0)

    norm_x = torch.sum(x ** 2, 1).unsqueeze(0)
    norm_y = torch.sum(y ** 2, 1).unsqueeze(0)

    ones_x = torch.ones(nx, 1).to(defaults.device)
    ones_y = torch.ones(ny, 1).to(defaults.device)

    a = torch.mm(ones_y, norm_x)
    b = torch.mm(x, y.t())
    c = torch.mm(ones_x, norm_y)

    return a.t() - 2 * b + c


def factorial(p):
    """
    Compute p!
    """
    return torch.prod(torch.tensor([i for i in range(1, p+1)], dtype=torch.double))


def factorial_frac(q, p):
    """
    Compute p!/q! for q <=p
    """
    if q > p:
        raise ValueError
    return torch.prod(torch.tensor([i for i in range(q+1, p+1)], dtype=torch.double))
