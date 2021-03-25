import torch
import torch.nn as nn

from src import default_params as defaults


def mnl_probabilities(features, w):
    """
    Compute the output of the softmax function given input features, w.
    """
    z = torch.mm(features, w)
    probs = nn.functional.softmax(z, dim=1)
    return probs


def mnl_gradient(probs, features, one_hot_labels, w, lam):
    """
    Compute the gradient of the multinomial loss.
    """
    n = features.shape[0]
    grad = - 1 / n * features.t().mm(one_hot_labels) + 1 / n * torch.mm(features.t(), probs)
    grad[1:] = grad[1:] + 2 * lam * w[1:]
    return grad


def mnl_hessian(features, probs, w, lam):
    """
    Compute the Hessian of the multinomial loss.
    """
    dim = w.nelement()
    d, k = w.shape
    hessian = torch.zeros((dim, dim), device=defaults.device)
    n = features.shape[0]
    id = torch.eye(d, device=defaults.device)
    id[0, 0] = 0

    # dL/dbeta_idbeta_i^T
    for i in range(k):
        hessian[i*d:(i+1)*d, i*d:(i+1)*d] = 1/n*(probs[:, i:i+1]*(1-probs[:, i:i+1])*features).t().mm(features) \
                                            + 2*lam*id

    # dL/dbeta_idbeta_j^T
    for i in range(k):
        for j in range(k):
            if i != j:
                hessian[i*d:(i+1)*d, j*d:(j+1)*d] = -1/n*(probs[:, i:i+1]*probs[:, j:j+1]*features).t().mm(features)
                hessian[j*d:(j+1)*d, i*d:(i+1)*d] = hessian[i*d:(i+1)*d, j*d:(j+1)*d]

    return hessian


def mnl_hessian_diag(features, probs, lam):
    """
    Compute the diagonal of the Hessian of the multinomial loss.
    """
    n, d = features.shape
    k = probs.shape[1]
    diag_hess = torch.mean(((probs*(1 - probs)).t().unsqueeze(2)*(features*features).unsqueeze(0)), 1).view(-1) + 2*lam
    for i in range(k):
        diag_hess[i*(d+1)] = diag_hess[i*(d+1)] - 2*lam

    return diag_hess
