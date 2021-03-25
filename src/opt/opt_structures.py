import os
import pickle
import torch
import torch.nn as nn


class Data:
    """
    Class that stores the dataloaders.
    """
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_iter = iter(self.train_loader)


class Model(nn.Module):
    """
    Class that stores a model and has methods to evaluate, save, and load it.
    """
    def __init__(self, model=None, save_path=None):
        super(Model, self).__init__()
        self.model = model
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def forward(self, x):
        return self.model(x)

    def save(self, **kwargs):
        if self.save_path is not None:
            save_dict = {'model': self.model}
            for key, value in kwargs.items():
                save_dict[key] = value
            torch.save(save_dict, self.save_path[:-7] + '_' + str(kwargs['iteration']) + self.save_path[-7:])

    def load(self, path):
        model_dict = torch.load(path)
        self.model = model_dict['model']
        return model_dict


class Params:
    """
    Class that stores the parameters for the optimization.
    """
    def __init__(self, num_classes=10, ckn=True, train_w_layers=None, lambda_filters=0, lambda_classifier=None,
                 standardize=False, normalize=False, w_last_init=None, opt_method='ulr-sgo',
                 update_step_size_method='fixed', step_size_init=None, step_size_schedule=None,
                 lambda_filters_schedule=None, tau=None, diag_hessian=False, maxiter_classifier=1000, num_iters=1000,
                 save_path=None, eval_test_every=10, save_every=100):
        self.num_classes = num_classes
        self.ckn = ckn
        self.train_w_layers = train_w_layers
        self.lambda_filters = lambda_filters
        self.lambda_classifier = lambda_classifier
        self.standardize = standardize
        self.normalize = normalize
        self.w_last_init = w_last_init
        self.opt_method = opt_method
        self.update_step_size_method = update_step_size_method
        self.step_size_init = step_size_init
        self.step_size_schedule = step_size_schedule
        self.lambda_filters_schedule = lambda_filters_schedule
        self.tau = tau
        self.diag_hessian = diag_hessian
        self.maxiter_classifier = maxiter_classifier
        self.num_iters = num_iters
        self.save_path = save_path
        self.eval_test_every = eval_test_every
        self.save_every = save_every

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def save(self):
        save_dir = os.path.join(*self.save_path.split(os.sep)[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value


class Results:
    """
    Class that stores the results and can update, save, and load them.
    """
    def __init__(self, save_path=None):
        self.save_path = save_path

        if self.save_path is not None:
            save_dir = os.path.join(*save_path.split(os.sep)[:-1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def update(self, iteration, **kwargs):
        for key, value in kwargs.items():
            if key not in self.__dict__:
                self.__dict__[key] = {}
            try:
                self.__dict__[key][iteration] = value.cpu().detach()
            except:
                try:
                    self.__dict__[key][iteration] = value.detach()
                except:
                    self.__dict__[key][iteration] = value

    def save(self):
        if self.save_path is not None:
            pickle.dump(self.__dict__, open(self.save_path, 'wb'))

    def load(self, path):
        params = pickle.load(open(path, 'rb'))
        for key, value in params.items():
            self.__dict__[key] = value
