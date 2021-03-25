import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

from src import default_params as defaults
from src.opt.opt_utils import compute_all_features


def next_lambda(accuracies, idxs):
    """
    Perform the next step of golden section search to get the next value of the validation parameter to try given the
    accuracies for the ones previously tried and the current interval under consideration.
    """
    tau = 0.5 + np.sqrt(5)/2
    x1 = idxs[0]
    x4 = idxs[-1]
    if idxs[0] is None:
        idxs[0] = 0
        return 0, idxs
    elif idxs[3] is None:
        idxs[-1] = len(accuracies)-1
        return -1, idxs
    elif idxs[1] is None:
        x2 = int((1/tau*x1 + (1-1/tau)*x4))
        idxs[1] = x2
        return x2, idxs
    elif idxs[2] is None:
        x3 = int(((1-1/tau)*x1 + (1/tau)*x4))
        idxs[2] = x3
        return x3, idxs
    else:
        if accuracies[idxs[1]] > accuracies[idxs[2]]:
            idxs[3] = idxs[2]
            idxs[2] = idxs[1]
            idxs[1] = int(np.rint((1/tau*idxs[2] + (1-1/tau)*idxs[0])))
            return idxs[1], idxs
        else:
            idxs[0] = idxs[1]
            idxs[1] = idxs[2]
            idxs[2] = int(np.rint((1/tau*idxs[1] + (1-1/tau)*idxs[3])))
            return idxs[2], idxs


def train(train_loader, valid_loader, test_loader, model, num_classes, maxiter, w_init=None, normalize=True,
          standardize=False, lambdas=None):
    """
    Generate features from the given model and then train a classifier. Perform hold-out validation for the
    regularization parameter lambda of the loss function.
    """
    found_best = False
    if lambdas is None:
        lambdas = [2 ** i for i in range(-40, 1, 1)]
    elif len(lambdas) == 1:
        best_lambda = lambdas[0]
        found_best = True

    accuracies = [None]*len(lambdas)
    idxs = [None]*4

    with torch.autograd.set_grad_enabled(False):
        x_train, x_valid, x_test, y_train, y_valid, y_test = compute_all_features(train_loader, valid_loader,
                                                                                  test_loader, model,
                                                                                  normalize=normalize,
                                                                                  standardize=standardize)

    model.cpu()
    x_train, x_valid, x_test = x_train.to(defaults.device), x_valid.to(defaults.device), x_test.to(defaults.device)
    y_train, y_valid, y_test = y_train.to(defaults.device), y_valid.to(defaults.device), y_test.to(defaults.device)

    # Hold-validation for lambda
    while found_best is False:
        lambda_idx, idxs = next_lambda(accuracies, idxs)
        if accuracies[lambda_idx] is None:
            valid_acc, train_acc, valid_loss, train_loss, w = train_classifier(x_train, x_valid, y_train, y_valid,
                                                                               lambdas[lambda_idx], num_classes,
                                                                               maxiter=maxiter, w=w_init)
            accuracies[lambda_idx] = valid_acc
        else:
            best_lambda = lambdas[lambda_idx]
            found_best = True

    # Final training with best lambda
    test_acc, train_acc, test_loss, train_loss, w = train_classifier(x_train, x_test, y_train, y_test, best_lambda,
                                                                     num_classes, maxiter=maxiter, w=w_init)
    valid_acc, valid_loss = compute_accuracy(x_valid, y_valid, w, torch.nn.CrossEntropyLoss())

    model.to(defaults.device)
    results_dict = {'train_accuracy': train_acc, 'valid_accuracy': valid_acc, 'test_accuracy': test_acc,
                    'train_loss': train_loss, 'valid_loss': valid_loss, 'test_loss': test_loss, 'w': w,
                    'lambda': best_lambda}

    return results_dict


def train_classifier(x_train, x_test, y_train, y_test, lam, num_classes, maxiter=1000, w=None):
    """
    Train a linear classifier on the training data and evaluate it on the given test data (if not None).
    """
    def obj(w):
        if w.__class__ == np.ndarray:
            w = w.reshape(x_train.shape[1] + 1, num_classes)
            w = torch.Tensor(w).to(defaults.device)
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        return obj_val.item()

    def grad(w):
        w = w.reshape(x_train.shape[1] + 1, num_classes)
        w = nn.Parameter(torch.Tensor(w).to(defaults.device))
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        obj_val.backward()
        grad = w.grad.data.detach().cpu().double().numpy().ravel()
        return grad

    if w is None:
        np.random.seed(0)
        w = np.random.normal(size=(np.size(x_train, 1) + 1, num_classes))

    loss = torch.nn.CrossEntropyLoss()
    if maxiter > 0:
        opt = scipy.optimize.minimize(obj, w.flatten(), method='L-BFGS-B', jac=grad,
                                      options={'maxiter': maxiter, 'disp': False})
        w = opt['x'].reshape(*w.shape)

    torch.autograd.set_grad_enabled(False)
    w = nn.Parameter(torch.Tensor(w).to(defaults.device))

    train_accuracy, train_loss = compute_accuracy(x_train, y_train, w, loss)
    test_accuracy, test_loss = compute_accuracy(x_test, y_test, w, loss)

    torch.autograd.set_grad_enabled(True)

    return test_accuracy, train_accuracy, test_loss, train_loss, w


def compute_accuracy(x, y, w, loss):
    """
    Given generated features x, weights w, true labels y, and a loss function, compute the value of the given loss
    function and the accuracy.
    """
    if x is not None:
        yhat = torch.mm(x, w[1:, :]) + w[0, :]
        loss_value = loss(yhat, y).item()
        yhat = torch.max(yhat, 1)[1]
        accuracy = np.mean((yhat == y).cpu().data.numpy())
    else:
        accuracy = None
        loss_value = None

    return accuracy, loss_value
