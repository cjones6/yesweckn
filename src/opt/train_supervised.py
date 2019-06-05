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
from __future__ import print_function
import copy
import numpy as np
import os
import time
import torch
import torch.nn as nn

from src import default_params as defaults
from . import train_unsupervised

if defaults.python3:
    import pickle
else:
    import cPickle as pickle


class TrainSupervised:
    """
    Class to perform supervised training of the filters of a CKN.
    """
    def __init__(self, train_loader, valid_loader, train_valid_loader, test_loader, model, nclasses, loss_name='mnl',
                 train_w_layers=None, wlast_init=None, diag_hessian=False, update_lr_method='line_search',
                 update_lr_freq=5, lr_init=None, nsteps_line_search=5, ls_range=(-3, 3), tau=None,
                 maxiter_wlast_full=1000, maxiter_outer=1000, lambdas=None, save_path=None, eval_test_every=10):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_valid_loader = train_valid_loader
        self.train_valid_iter = iter(self.train_valid_loader)
        self.test_loader = test_loader
        self.model = model
        self.nclasses = nclasses
        self.loss_name = loss_name
        self.train_w_layers = train_w_layers
        self.wlast = wlast_init
        self.diag_hessian = diag_hessian
        self.update_lr_method = update_lr_method
        self.update_lr_freq = update_lr_freq
        self.lr_init = lr_init
        self.nsteps_line_search = nsteps_line_search
        self.ls_range = ls_range
        self.tau = tau
        self.maxiter_wlast_full = maxiter_wlast_full
        self.maxiter_outer = maxiter_outer
        self.lambdas = lambdas
        self.save_path = save_path
        self.eval_test_every = eval_test_every
        self.iteration = 0
        # Dictionaries to store the results
        self.train_accuracies = {}
        self.valid_accuracies = {}
        self.test_accuracies = {}
        self.train_losses = {}
        self.test_losses = {}
        self.iter_times = {}
        self.step_sizes = []

        if self.loss_name == 'mnl':
            self.loss = torch.nn.CrossEntropyLoss()
        elif self.loss_name == 'square':
            self.loss = train_unsupervised.square_loss
        else:
            raise NotImplementedError

        if self.train_w_layers is None:
            self.train_w_layers = range(len(self.model.layers))

        if save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def _update_results(self, train_accuracy=None, valid_accuracy=None, test_accuracy=None, train_loss=None,
                        test_loss=None, iter_time=None):
        """
        Update the dictionaries storing the results from each iteration.
        """
        if train_accuracy is not None:
            self.train_accuracies[self.iteration] = train_accuracy
        if valid_accuracy is not None:
            self.valid_accuracies[self.iteration] = valid_accuracy
        if test_accuracy is not None:
            self.test_accuracies[self.iteration] = test_accuracy
        if train_loss is not None:
            self.train_losses[self.iteration] = train_loss
        if test_loss is not None:
            self.test_losses[self.iteration] = test_loss
        if iter_time is not None:
            self.iter_times[self.iteration] = iter_time

    def _get_batch(self):
        """
        Get a batch of training data (x, y, one-hot version of y) from the training + validation sets.
        """
        try:
            x_train, y_train = next(self.train_valid_iter)
        except:
            self.train_valid_iter = iter(self.train_valid_loader)
            x_train, y_train = next(self.train_valid_iter)
        y_train_one_hot = train_unsupervised.one_hot_embedding(y_train, self.nclasses)
        x_train = x_train.type(torch.get_default_dtype()).to(defaults.device)
        y_train = y_train.to(defaults.device)
        y_train_one_hot = y_train_one_hot.to(defaults.device)

        return x_train, y_train, y_train_one_hot

    def _compute_features(self, x_train):
        """
        Compute the features of a CKN on a minibatch of inputs.
        """
        features = self.model(x_train.to(defaults.device))
        features = features.contiguous().view(features.shape[0], -1)

        mean = torch.mean(features, 1)
        features = features - mean.unsqueeze(1)
        nrm = torch.mean(torch.norm(features, 2, 1))
        features = features / nrm

        return features

    def _compute_normalizations(self):
        """
        Compute the term k(W^TW)^{-1/2} for each layer.
        """
        for layer_num in range(len(self.model.layers)):
            self.model.layers[layer_num].store_normalization = True
            with torch.autograd.set_grad_enabled(False):
                self.model.layers[layer_num].normalization = self.model.layers[layer_num].compute_normalization()

    @staticmethod
    def _mnl_probabilities(features, w):
        """
        Compute the output of the softmax function given input features*w.
        """
        z = torch.mm(features, w)
        probs = nn.functional.softmax(z, dim=1)
        return probs

    @staticmethod
    def _mnl_gradient(probs, features, one_hot_labels, w, lam):
        """
        Compute the gradient of the multinomial loss.
        """
        n = features.shape[0]
        grad = - 1/n*features.t().mm(one_hot_labels) + 1/n*torch.mm(features.t(), probs)
        grad[1:] = grad[1:] + 2 * lam * w[1:]
        return grad

    @staticmethod
    def _mnl_hessian(features, probs, w, lam):
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
            hessian[i*d:(i+ 1)*d, i*d:(i+1)*d] = 1/n*(probs[:, i:i+1]*(1-probs[:, i:i+1])*features).t().mm(features) \
                                                 + 2*lam*id

        # dL/dbeta_idbeta_j^T
        for i in range(k):
            for j in range(k):
                if i != j:
                    hessian[i*d:(i+1)*d, j*d:(j+1)*d] = -1/n*(probs[:, i:i+1]*probs[:, j:j+1]*features).t().mm(features)
                    hessian[j*d:(j+1)*d, i*d:(i+1)*d] = hessian[i*d:(i+1)*d, j*d:(j+1)*d]

        return hessian

    @staticmethod
    def _mnl_hessian_diag(features, probs, lam):
        """
        Compute the diagonal of the Hessian of the multinomial loss.
        """
        n, d = features.shape
        k = probs.shape[1]
        diag_hess = torch.mean(((probs * (1 - probs)).t().unsqueeze(2) * (features * features).unsqueeze(0)), 1).view(-1) + 2 * lam
        for i in range(k):
            diag_hess[i * (d + 1)] = diag_hess[i * (d + 1)] - 2 * lam

        return diag_hess

    def _get_step_sizes_nsteps(self):
        """
        Get the step sizes to try at this iteration and the number of steps to take (1 if not doing line search;
        self.nsteps_line_search otherwise).
        """
        init_step_size_filters = None if len(self.step_sizes) == 0 else self.step_sizes[-1]
        if self.iteration % self.update_lr_freq == 0 or self.iteration == 1:
            if 'line_search' in self.update_lr_method:
                if init_step_size_filters is None:
                    init_step_size_filters = 2 ** -3
                    all_step_sizes_filters = [2 ** i * init_step_size_filters for i in range(-6, 3)]
                else:
                    all_step_sizes_filters = [2 ** i * init_step_size_filters for i in
                                          range(self.ls_range[0], self.ls_range[1]+1)]
            elif 'fixed' in self.update_lr_method:
                all_step_sizes_filters = [self.lr_init]
            else:
                raise NotImplementedError
            nsteps = self.nsteps_line_search
        else:
            all_step_sizes_filters = [init_step_size_filters]
            nsteps = 1

        return all_step_sizes_filters, nsteps

    def _update_filters(self):
        """
        Take one step (if not updating the learning rate. nsteps_line_search otherwise.) to optimize the filters at each
        layer.
        """
        all_step_sizes_filters, nsteps = self._get_step_sizes_nsteps()
        x_train, y_train, y_train_one_hot = self._get_batch()
        x_train2, y_train2, y_train_one_hot2 = self._get_batch()

        for layer_num in range(len(self.model.layers)):
            self.model.layers[layer_num].store_normalization = False

        if len(all_step_sizes_filters) > 1:
            best_step_size = -1
            best_obj = np.inf
            model_old = copy.deepcopy(self.model.state_dict())
            wlast_old = self.wlast.clone()

        for step_size_filters in all_step_sizes_filters:
            for i in range(nsteps):
                torch.set_grad_enabled(True)
                self.model.zero_grad()
                obj_value, _ = self._ultimate_layer_reversal(x_train, y_train, y_train_one_hot, tau=self.tau)
                obj_value.backward()
                with torch.autograd.set_grad_enabled(False):
                    for layer_num in self.train_w_layers:
                        grad = self.model.layers[layer_num].W.grad.data
                        W = self.model.layers[layer_num].W.data
                        W = W - step_size_filters * grad / torch.norm(grad, 2, 1, keepdim=True)
                        W_proj = W / torch.norm(W, 2, 1, keepdim=True)
                        self.model.layers[layer_num].W.data = W_proj
                        if self.loss_name != 'square':
                            _, eta = self._ultimate_layer_reversal(x_train, y_train, y_train_one_hot, tau=self.tau)
                if self.loss_name != 'square':
                    self.wlast = self.wlast + eta
            if len(all_step_sizes_filters) > 1:
                obj_value = self._optimize_wlast_batch(x_train2, y_train2)
                if obj_value < best_obj:
                    best_obj = obj_value
                    best_step_size = step_size_filters
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_wlast = self.wlast.clone()

                self.model.load_state_dict(model_old)
                self.wlast = wlast_old.clone()
            else:
                best_step_size = step_size_filters

        if len(all_step_sizes_filters) > 1:
            self.wlast = best_wlast.clone()
            self.model.load_state_dict(best_model)
        self.iteration += nsteps-1
        self.step_sizes.append(best_step_size)

    def _ultimate_layer_reversal(self, x_train, y_train, y_train_one_hot, tau=None):
        """
        Compute the objective function and step eta of the ultimate layer reversal method.
        """
        features = self._compute_features(x_train)
        n, d = features.shape

        if self.loss_name == 'square':
            pi = torch.eye(n, device=defaults.device) - 1.0/n*torch.ones(n, n, device=defaults.device)
            xpiy = features.t().mm(pi).mm(y_train_one_hot)
            inv_term = features.t().mm(pi).mm(features) + n*self.lam*torch.eye(d, device=defaults.device)
            wlast, _ = torch.gesv(xpiy, inv_term)
            obj = 1/n*(torch.sum((pi.mm(y_train_one_hot))**2) - torch.trace(xpiy.t().mm(wlast)))

            return obj, None

        elif self.loss_name == 'mnl':
            self.wlast = self.wlast.detach()
            features = torch.cat((torch.ones(n, 1, device=defaults.device), features), 1)
            probs = self._mnl_probabilities(features, self.wlast)
            grad_wlast = self._mnl_gradient(probs, features, y_train_one_hot, self.wlast, self.lam)
            grad_wlast = grad_wlast.t().contiguous().view(-1)

            if not self.diag_hessian:
                hessian_wlast = self._mnl_hessian(features, probs, self.wlast, self.lam)
                if len(grad_wlast) > 65*10:  # If it's too small it's faster to run it on the CPU
                    eta = -torch.gesv(grad_wlast.unsqueeze(1), hessian_wlast + tau * torch.eye(hessian_wlast.shape[0],
                                                                                device=defaults.device))[0].squeeze()
                else:
                    eta = -torch.gesv(grad_wlast.unsqueeze(1).cpu(), hessian_wlast.cpu() +
                                      tau * torch.eye(hessian_wlast.shape[0]))[0].squeeze().to(defaults.device)
                hess_term = 0.5 * eta.dot(hessian_wlast.mv(eta))
            else:
                hessian_wlast = self._mnl_hessian_diag(features, probs, self.lam)
                eta = 1 / (hessian_wlast + tau) * grad_wlast
                hess_term = 0.5 * eta.dot(hessian_wlast * eta)

            obj = self.loss(torch.mm(features, self.wlast), y_train) + self.lam * torch.norm(self.wlast[1:]) ** 2 \
                  + grad_wlast.dot(eta) + hess_term \
                  + tau / 2 * torch.sum(eta ** 2)

            return obj, eta.view(self.nclasses, -1).t()
        else:
            raise NotImplementedError

    def _optimize_wlast_batch(self, x_train, y_train):
        """
        Optimize the parameters of the classifier on a minibatch.
        """
        with torch.autograd.set_grad_enabled(False):
            features = self._compute_features(x_train)
        loss = train_unsupervised.train_classifier(features, None, y_train, None, self.lam, self.nclasses,
                                                   maxiter=self.maxiter_wlast_full, w=self.wlast.cpu().data.numpy(),
                                                   loss_name=self.loss_name)[3]

        return loss

    def _optimize_classifier_full(self, update_wlast=False):
        """
        Optimize the classifier on the full dataset, updating the parameters of the classifier afterward if specified.
        """
        self._compute_normalizations()
        if self.wlast.__class__ == torch.nn.parameter.Parameter or self.wlast.__class__ == torch.Tensor:
            wlast = self.wlast.data.cpu().numpy()
        else:
            wlast = self.wlast
        results = train_unsupervised.train(self.train_loader, self.valid_loader, self.test_loader, self.model,
                                           self.nclasses, self.maxiter_wlast_full, wlast, normalize=True,
                                           standardize=False, loss_name=self.loss_name, lambdas=self.lambdas)
        test_acc, valid_acc, train_acc, test_loss, train_loss, w, best_lambda = results
        if self.iteration == 0:
            print('Iteration \t Test accuracy \t Train accuracy \t Test loss \t Train loss \t Reg param')
        print(self.iteration, '\t\t {:06.4f}'.format(test_acc), '\t\t', '{:06.4f}'.format(train_acc), '\t',
              '{:06.4f}'.format(test_loss), '\t', '{:06.4f}'.format(train_loss), '\t', '{:.2e}'.format(best_lambda))
        if update_wlast:
            self.wlast = w
        self.lambdas = [best_lambda]
        self.lam = best_lambda
        self._update_results(train_accuracy=train_acc, valid_accuracy=valid_acc, test_accuracy=test_acc,
                             train_loss=train_loss, test_loss=test_loss)

    def train(self):
        """
        Train a CKN in a supervised manner using the ultimate layer reversal method.
        """
        # Initial training of classifier
        self._optimize_classifier_full(update_wlast=True)
        iter_since_last_eval = 0

        # Alternating optimization
        while self.iteration < self.maxiter_outer:
            prev_iteration = self.iteration
            self.iteration += 1
            t1 = time.time()
            self._update_filters()
            t2 = time.time()
            self._update_results(iter_time=t2-t1)
            iter_since_last_eval += self.iteration - prev_iteration

            if iter_since_last_eval >= self.eval_test_every:
                if self.loss_name == 'square':
                    update_w = True
                else:
                    update_w = False
                self._optimize_classifier_full(update_wlast=update_w)
                self.train_valid_iter = None
                train_dict = copy.deepcopy(self.__dict__)
                del train_dict['train_loader']
                del train_dict['valid_loader']
                del train_dict['train_valid_loader']
                del train_dict['test_loader']
                pickle.dump(train_dict, open(self.save_path + '_model_iter_' + str(self.iteration) + '.pickle', 'wb'))

                pickle.dump({'test_acc': self.test_accuracies, 'train_acc': self.train_accuracies,
                             'valid_acc': self.valid_accuracies, 'test_loss': self.test_losses,
                             'train_loss': self.train_losses, 'lambda': self.lam, 'wlast': self.wlast,
                             'step_sizes': self.step_sizes},
                           open(self.save_path + '_results.pickle', 'wb'))
                iter_since_last_eval = 0

            if self.iteration > self.maxiter_outer:
                break

        print('Optimizing classifier on full training set')
        self.lambdas = None
        self._optimize_classifier_full(update_wlast=True)
        pickle.dump({'test_acc': self.test_accuracies, 'train_acc': self.train_accuracies,
                     'valid_acc': self.valid_accuracies, 'test_loss': self.test_losses,
                     'train_loss': self.train_losses, 'lambda': self.lam, 'wlast': self.wlast,
                     'step_sizes': self.step_sizes},
                    open(self.save_path + '_results.pickle', 'wb'))

        self.train_valid_iter = None
        train_dict = copy.deepcopy(self.__dict__)
        del train_dict['train_loader']
        del train_dict['valid_loader']
        del train_dict['train_valid_loader']
        del train_dict['test_loader']
        pickle.dump(train_dict, open(self.save_path + '_model.pickle', 'wb'))
