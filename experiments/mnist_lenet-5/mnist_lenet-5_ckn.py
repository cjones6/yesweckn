"""
Train a LeNet-5 CKN on the MNIST dataset
"""

import argparse
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn

sys.path.append('../..')

from src import default_params as defaults
import src.data_loaders.mnist as mnist
from src.model.ckn import net, parse_config
from src.opt import opt_structures, train_network

# Parameters for the model, data, and training
parser = argparse.ArgumentParser(description='LeNet-5 CKN training on the MNIST dataset')
parser.add_argument('--batch_size', default=4096, type=int,
                    help='Batch size for training')
parser.add_argument('--data_path', default='../../data/mnist', type=str,
                    help='Location of the MNIST dataset')
parser.add_argument('--eval_test_every', default=25, type=int,
                    help='Evaluate the performance every x iterations')
parser.add_argument('--gpu', default='0', type=str,
                    help='Which GPUs to use')
parser.add_argument('--hessian_reg', default=-7, type=int,
                    help='log2(Regularization of the Hessian for the ULR-SGO method)')
parser.add_argument('--num_filters', default=32, type=int,
                    help='Number of filters per layer')
parser.add_argument('--num_iters', default=10000, type=int,
                    help='Number of total iterations to perform')
parser.add_argument('--opt_method', default='ulr-sgo', type=str,
                    help="Optimization method to use. Either 'ulr-sgo' or 'sgo'.")
parser.add_argument('--save_path', default='../../results/mnist_lenet-5_ckn', type=str,
                    help='Directory where the model, parameters, and results will be stored')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--step_size', default=-5, type=int,
                    help='log2(Initial step size)')
parser.add_argument('--step_size_method', default='fixed', type=str,
                    help='How to vary the step size')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.hessian_reg = 2**args.hessian_reg
args.step_size = 2**args.step_size
bw = 0.6
num_classes = 10
print(args)

save_dir = args.save_path
save_file = os.path.join(save_dir, str(bw) + '_' + str(args.hessian_reg) + '_' + str(args.num_filters)
                         + '_' + str(args.opt_method) + '_' + str(args.seed) + '_' + str(args.step_size)
                         + '_' + str(time.time()))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create the data loaders
train_loader, valid_loader, test_loader = mnist.get_dataloaders(batch_size=args.batch_size, data_path=args.data_path,
                                                                num_workers=4)


# Load and initialize the model
params = parse_config.load_config('../../cfg/lenet-5_ckn.cfg')
if args.num_filters > 0:
    num_layers = len(params['num_filters'])
    params['num_filters'] = [args.num_filters] * num_layers
    params['patch_sigma'] = [bw] * num_layers

layers = parse_config.create_layers(params)
model = net.CKN(layers)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model).to(defaults.device)
    model.module.init(train_loader)
else:
    model.to(defaults.device).init(train_loader)
print('Done with the initialization')


# Create the step size schedule
if args.step_size_method == 'fixed':
    step_size_schedule = None
    update_step_size_method = 'fixed'
elif args.step_size_method == 'best-ckn-128':
    def step_size_schedule(x):
        if x < 2000:
            return 2 ** -7
        elif 2000 <= x < 4000:
            return 2 ** -7
        elif 4000 <= x < 6000:
            return 2 ** -8
        elif 6000 <= x < 8000:
            return 2 ** -7
        elif 8000 <= x:
            return 2 ** -7
    update_step_size_method = 'schedule'
elif args.step_size_method == 'best-ckn-64':
    def step_size_schedule(x):
        if x < 2000:
            return 2 ** -6
        elif 2000 <= x < 4000:
            return 2 ** -8
        elif 4000 <= x < 6000:
            return 2 ** -8
        elif 6000 <= x < 8000:
            return 2 ** -10
        elif 8000 <= x:
            return 2 ** -10
    update_step_size_method = 'schedule'
elif args.step_size_method == 'best-ckn-32':
    def step_size_schedule(x):
        if x < 2000:
            return 2 ** -5
        elif 2000 <= x < 4000:
            return 2 ** -6
        elif 4000 <= x < 6000:
            return 2 ** -5
        elif 6000 <= x < 8000:
            return 2 ** -5
        elif 8000 <= x:
            return 2 ** -5
    update_step_size_method = 'schedule'
elif args.step_size_method == 'best-ckn-16':
    def step_size_schedule(x):
        # pass
        if x < 2000:
            return 2 ** -8
        elif 2000 <= x < 4000:
            return 2 ** -7
        elif 4000 <= x < 6000:
            return 2 ** -8
        elif 6000 <= x < 8000:
            return 2 ** -8
        elif 8000 <= x:
            return 2 ** -9
    update_step_size_method = 'schedule'
elif args.step_size_method == 'best-ckn-8':
    def step_size_schedule(x):
        if x < 2000:
            return 2 ** -7
        elif 2000 <= x < 4000:
            return 2 ** -8
        elif 4000 <= x < 6000:
            return 2 ** -7
        elif 6000 <= x < 8000:
            return 2 ** -6
        elif 8000 <= x:
            return 2 ** -6
    update_step_size_method = 'schedule'
else:
    raise NotImplementedError


# Set up the data, parameters, model, optimizer, and results objects
data = opt_structures.Data(train_loader, valid_loader, test_loader)
params = opt_structures.Params(num_classes=num_classes, ckn=True, train_w_layers=[0, 2, 4, 5], lambda_filters=0,
                               normalize=True, opt_method=args.opt_method, w_last_init=None,
                               update_step_size_method=update_step_size_method, step_size_init=args.step_size,
                               step_size_schedule=step_size_schedule, tau=args.hessian_reg, num_iters=args.num_iters,
                               save_path=save_file + '_params.pickle', eval_test_every=args.eval_test_every,
                               save_every=500)
model = opt_structures.Model(model, save_path=save_file + '_model.pickle')
results = opt_structures.Results(save_path=save_file + '_results.pickle')
optimizer = train_network.TrainSupervised(data, model, params, results)


# Train the model
optimizer.train()
