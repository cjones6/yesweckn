import time
import torch
import torch.nn as nn
import torch.optim as optim

from src import default_params as defaults
from . import train_classifier, opt_utils, ulr_utils


class TrainSupervised:
    """
    Class to perform supervised training of the parameters of a network.
    """
    def __init__(self, data, model, params, results):
        self.data = data
        self.model = model
        self.params = params
        self.results = results
        self.loss = torch.nn.CrossEntropyLoss()

        # Initialize things that get updated during the training
        self.iteration = 0
        self.w_last = params.w_last_init
        self.w_prev = None

        if hasattr(self.model.model, 'layers'):
            self.distributed = False
        else:
            self.distributed = True

        if self.params.train_w_layers is None and self.params.ckn:
            if self.distributed:
                self.params.train_w_layers = range(len(self.model.model.module.layers))
            else:
                self.params.train_w_layers = range(len(self.model.model.layers))

        if self.params.opt_method == 'ulr-sgo' and not self.params.ckn:
            self.optimizer = optim.SGD(self.model.model.parameters(), lr=self.params.step_size_init,
                                       weight_decay=self.params.lambda_filters)
        elif self.params.opt_method == 'sgo':
            self.w_last = nn.Parameter(self.w_last)
            params = list(self.model.model.parameters())
            params.append(self.w_last)
            self.optimizer = optim.SGD(params, lr=self.params.step_size_init, weight_decay=self.params.lambda_filters)

    def _get_step_size(self):
        """
        Get the step size to use at this iteration.
        """
        if self.params.update_step_size_method == 'schedule' and self.params.ckn:
            return self.params.step_size_schedule(self.iteration)
        elif self.params.update_step_size_method == 'schedule':
            if self.iteration == 0 or \
                  (self.params.step_size_schedule(self.iteration) != self.params.step_size_schedule(self.iteration-1) or
                   self.params.lambda_filters_schedule(self.iteration) !=
                   self.params.lambda_filters_schedule(self.iteration-1)):

                self.optimizer = optim.SGD(self.model.model.parameters(),
                                           lr=self.params.step_size_schedule(self.iteration),
                                           weight_decay=self.params.lambda_filters_schedule(self.iteration))
                print('Updated optimizer with lr, wd:', self.params.step_size_schedule(self.iteration),
                                                        self.params.lambda_filters_schedule(self.iteration))
            return self.params.step_size_schedule(self.iteration)
        else:
            return self.params.step_size_init

    def _update_filters(self):
        """
        Take one step to optimize the filters at each layer.
        """
        self.model.train()
        step_size = self._get_step_size()
        x, y = opt_utils.get_batch(self.data)

        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        self.model.zero_grad()
        if self.params.opt_method == 'sgo':
            self.optimizer.zero_grad()
        if self.params.opt_method == 'ulr-sgo':
            obj_value, _ = self._ultimate_layer_reversal(x, y)
        else:
            obj_value = self._sgo_step(x, y)
        torch.cuda.empty_cache()
        obj_value.backward()

        if self.params.ckn:
            with torch.autograd.set_grad_enabled(False):
                for layer_num in self.params.train_w_layers:
                    if not self.distributed:
                        grad = self.model.model.layers[layer_num].W.grad.data
                        W = self.model.model.layers[layer_num].W.data
                        W = W - step_size*grad/torch.norm(grad, 2, 1, keepdim=True)
                        W_proj = W / torch.norm(W, 2, 1, keepdim=True)
                        self.model.model.layers[layer_num].W.data = W_proj
                    else:
                        grad = self.model.model.module.layers[layer_num].W.grad.data
                        W = self.model.model.module.layers[layer_num].W.data
                        W = W - step_size * grad / torch.norm(grad, 2, 1, keepdim=True)
                        W_proj = W / torch.norm(W, 2, 1, keepdim=True)
                        self.model.model.module.layers[layer_num].W.data = W_proj
                if self.params.opt_method == 'sgo':
                    self.w_last.data = self.w_last.data-step_size*self.w_last.grad.data
        else:
            self.optimizer.step()

        if self.params.opt_method == 'ulr-sgo':
            _, eta = self._ultimate_layer_reversal(x, y)
            self.w_last = self.w_last + eta

        self.results.update(self.iteration, step_size=step_size)

    def _ultimate_layer_reversal(self, x, y):
        """
        Compute the value of the approximate objective used in the ultimate layer reversal method and update W, b.
        """
        features = opt_utils.compute_features(x, self.model, normalize=self.params.normalize,
                                              standardize=self.params.standardize)
        y_one_hot = opt_utils.one_hot_embedding(y, self.params.num_classes)

        n = len(features)
        self.w_last = self.w_last.detach()
        features = torch.cat((torch.ones(n, 1, device=defaults.device), features), 1)

        probs = ulr_utils.mnl_probabilities(features, self.w_last)
        grad_wlast = ulr_utils.mnl_gradient(probs, features, y_one_hot, self.w_last, self.params.lambda_classifier)
        grad_wlast = grad_wlast.t().contiguous().view(-1)
        if not self.params.diag_hessian:
            hessian_wlast = ulr_utils.mnl_hessian(features, probs, self.w_last, self.params.lambda_classifier)
            if len(grad_wlast) > 65 * 10:  # If it's too small it's faster to run it on the CPU
                eta = - torch.solve(grad_wlast.unsqueeze(1), hessian_wlast + self.params.tau *
                                    torch.eye(hessian_wlast.shape[0], device=defaults.device))[0].squeeze()
            else:
                eta = -torch.solve(grad_wlast.unsqueeze(1).cpu(), hessian_wlast.cpu() +
                                   self.params.tau * torch.eye(hessian_wlast.shape[0]))[0].squeeze().to(defaults.device)
            hess_term = 0.5 * eta.dot(hessian_wlast.mv(eta))
        else:
            hessian_wlast = ulr_utils.mnl_hessian_diag(features, probs, self.params.lambda_classifier)
            eta = 1/(hessian_wlast + self.params.tau) * grad_wlast
            hess_term = 0.5 * eta.dot(hessian_wlast * eta)

        obj = self.loss(torch.mm(features, self.w_last), y) \
              + self.params.lambda_classifier*torch.norm(self.w_last[1:])**2 \
              + grad_wlast.dot(eta) + hess_term \
              + self.params.tau/2 * torch.sum(eta ** 2)

        eta = eta.view(self.params.num_classes, -1).t()
        self.model.train()

        return obj, eta

    def _sgo_step(self, x, y):
        """
        Compute the value of the objective.
        """
        features = opt_utils.compute_features(x, self.model, normalize=self.params.normalize,
                                              standardize=self.params.standardize)
        features = torch.cat((torch.ones(len(y), 1, device=defaults.device), features), 1)
        obj = self.loss(torch.mm(features, self.w_last), y)

        return obj

    def _optimize_classifier_full(self):
        """
        Optimize the classifier on the full dataset.
        """
        self.model.eval()
        if self.params.ckn and not self.distributed:
            self.model.model = opt_utils.compute_normalizations(self.model.model)

        lambdas = [self.params.lambda_classifier] if self.iteration != 0 else None
        results = train_classifier.train(self.data.train_loader, self.data.valid_loader, self.data.test_loader,
                                         self.model, self.params.num_classes, self.params.maxiter_classifier,
                                         w_init=self.w_prev, normalize=self.params.normalize,
                                         standardize=self.params.standardize, lambdas=lambdas)
        self.w_prev = results['w'].detach().cpu()

        if self.iteration == 0:
            self.params.lambda_classifier = results['lambda']
            if self.params.opt_method == 'sgo':
                self.w_last.data = results['w']
            else:
                self.w_last = results['w']

        opt_utils.print_results(self.iteration, results, header=(self.iteration == 0))
        self.results.update(self.iteration, **results)

        if self.params.ckn and not self.distributed:
            for layer_num in range(len(self.model.model.layers)):
                self.model.model.layers[layer_num].store_normalization = False
        self.model.train()

    def train(self):
        """
        Train a network in a supervised manner using the ultimate layer reversal method.
        """
        iter_since_last_eval = iter_since_last_save = 0
        self._optimize_classifier_full()

        while self.iteration < self.params.num_iters:
            t1 = time.time()
            self._update_filters()
            self.iteration += 1
            iter_since_last_eval += 1
            iter_since_last_save += 1

            if iter_since_last_eval >= self.params.eval_test_every:
                self._optimize_classifier_full()
                if iter_since_last_save >= self.params.save_every:
                    if self.params.ckn:
                        self.model.save(iteration=self.iteration, w_last=self.w_last)
                    else:
                        self.model.save(iteration=self.iteration, w_last=self.w_last, optimizer=self.optimizer)
                    self.results.save()
                    self.params.save()
                    iter_since_last_save = 0
                iter_since_last_eval = 0

            t2 = time.time()
            self.results.update(self.iteration, time=t2-t1)

        print('Done learning the features. Saving final results.')
        self._optimize_classifier_full()
        if self.params.ckn:
            self.model.save(iteration=self.iteration, w_last=self.w_last)
        else:
            self.model.save(iteration=self.iteration, w_last=self.w_last, optimizer=self.optimizer)
        self.results.save()
        self.params.save()
