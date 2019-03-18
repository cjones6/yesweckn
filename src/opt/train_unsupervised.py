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
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

from src import default_params as defaults


def next_lambda(accuracies, idxs):
    """
    Perform the next step of golden section search to get the next value of the cross-validation parameter to try given
    the accuracies for the ones previously tried and the current interval under consideration.
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


def train(train_loader, valid_loader, test_loader, model, nclasses, maxiter, w_init=None, normalize=True,
          standardize=False, loss_name='mnl', lambdas=None):
    """
    Generate features from the given model and then train a classifier. Perform cross-validation for the regularization
    parameter lambda of the loss function.
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
        x_train, x_valid, x_test, y_train, y_valid, y_test = generate_features(train_loader, valid_loader, test_loader,
                                                                    model, normalize=normalize, standardize=standardize)

    model.cpu()
    x_train, x_valid, x_test = x_train.to(defaults.device), x_valid.to(defaults.device), x_test.to(defaults.device)
    y_train, y_valid, y_test = y_train.to(defaults.device), y_valid.to(defaults.device), y_test.to(defaults.device)

    if valid_loader is not None:
        # Cross-validation over lambda
        while found_best is False:
            lambda_idx, idxs = next_lambda(accuracies, idxs)
            if accuracies[lambda_idx] is None:
                valid_acc, train_acc, valid_loss, train_loss, w = train_classifier(x_train, x_valid, y_train, y_valid,
                                                                                   lambdas[lambda_idx], nclasses,
                                                                                   maxiter=maxiter,
                                                                                   w=w_init, loss_name=loss_name)

                accuracies[lambda_idx] = valid_acc

            else:
                best_lambda = lambdas[lambda_idx]
                found_best = True
    else:
        raise NotImplementedError

    # Final training with best lambda
    x_train_val = torch.cat([x_train, x_valid])
    y_train_val = torch.cat([y_train, y_valid])
    accuracies = np.array(accuracies, dtype=np.float64)
    if not np.alltrue(np.isnan(accuracies)):
        valid_acc = np.nanmax(accuracies)
    else:
        valid_acc = np.nan

    acc, train_acc, test_loss, train_loss, w = train_classifier(x_train_val, x_test, y_train_val, y_test, best_lambda,
                                                                    nclasses, maxiter=maxiter, w=w_init,
                                                                    loss_name=loss_name)

    model.to(defaults.device)

    return acc, valid_acc, train_acc, test_loss, train_loss, w, best_lambda


def generate_features(train_loader, valid_loader, test_loader, model, normalize=True, standardize=False):
    """
    Generate features for all images in the training, validation, and test sets using the given model. Then normalize
    or standardize them.
    """
    dataset = {'train': {'x': [], 'y': []}, 'valid': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}}
    for dataset_name, data_loader in zip(sorted(dataset.keys()), [test_loader, train_loader, valid_loader]):
        nimages = 0
        for i, (x, y) in enumerate(data_loader):
            batch_size = len(y)
            nimages += batch_size
            x = x.type(torch.get_default_dtype()).to(defaults.device)
            features = model(x)
            features = features.contiguous().view(features.shape[0], -1).data.cpu()
            dataset[dataset_name]['x'].append(features)
            dataset[dataset_name]['y'].append(y)

        dataset[dataset_name]['x'] = torch.cat(dataset[dataset_name]['x'])
        dataset[dataset_name]['y'] = torch.cat(dataset[dataset_name]['y'])

    if normalize:
        for key in dataset.keys():
            dataset[key]['x'].sub_(torch.mean(dataset[key]['x'], 1, keepdim=True))
        nrm = torch.mean(torch.norm(dataset['train']['x'], 2, 1))
        for key in dataset.keys():
            dataset[key]['x'].div_(nrm)
    elif standardize:
        mean = torch.mean(dataset['train']['x'], 0, keepdim=True)
        sd = torch.std(dataset['train']['x'], 0, keepdim=True)
        for key in dataset.keys():
            dataset[key]['x'].sub_(mean)
            dataset[key]['x'].div_(sd)

    return dataset['train']['x'], dataset['valid']['x'], dataset['test']['x'], \
           dataset['train']['y'], dataset['valid']['y'], dataset['test']['y']


def train_classifier(x_train, x_test, y_train, y_test, lam, nclasses, maxiter=1000, w=None, loss_name='mnl'):
    """
    Train a linear classifier on the training data and evaluate it on the given test data (if not None).
    """
    def obj(w):
        if w.__class__ == np.ndarray:
            w = w.reshape(x_train.shape[1] + 1, nclasses)
            w = torch.Tensor(w).to(defaults.device)
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        return obj_val.item()

    def grad(w):
        w = w.reshape(x_train.shape[1] + 1, nclasses)
        w = nn.Parameter(torch.Tensor(w).to(defaults.device))
        yhat = torch.mm(x_train, w[1:, :]) + w[0, :]
        obj_val = loss(yhat, y_train) + lam * torch.norm(w[1:, :]) ** 2
        obj_val.backward()
        grad = w.grad.data.detach().cpu().double().numpy().ravel()
        return grad

    if w is None:
        w = np.random.normal(size=(np.size(x_train, 1) + 1, nclasses))

    if loss_name == 'square':
        y_train = one_hot_embedding(y_train, nclasses).type(torch.get_default_dtype()).to(defaults.device)
        if x_test is not None:
            y_test = one_hot_embedding(y_test, nclasses).type(torch.get_default_dtype()).to(defaults.device)
        loss = square_loss
    elif loss_name == 'mnl':
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

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
        if hasattr(loss, '__name__') and loss.__name__ == 'square_loss':
            y = torch.max(y, 1)[1]
        accuracy = np.mean((yhat == y).cpu().data.numpy())
    else:
        accuracy = None
        loss_value = None

    return accuracy, loss_value


def one_hot_embedding(y, n_dims):
    """
    Generate a one-hot representation of the input vector y.
    https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
    """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape[0], -1)

    return y_one_hot


def square_loss(y, yhat):
    """
    Compute the square loss given the true and predicted labels.
    """
    return 1/len(y)*torch.sum((y-yhat)**2)
