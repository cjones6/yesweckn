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

"""
Train the LeNet-1 CKN on MNIST
"""

from __future__ import division
import argparse
import matplotlib.pyplot as plt
import os
import cPickle as pickle
import sys

sys.path.append('..')

from src import default_params as defaults
from src.opt import train_supervised
from src.model import net, parse_config
import src.data_loaders.mnist as mnist

# Inputs for the data, model, and training
parser = argparse.ArgumentParser(description='LeNet-1 CKN training on MNIST')
parser.add_argument('--data_path', default='../data/MNIST', type=str,
                    help="Location of the MNIST dataset. If it doesn't exist it will be automatically downloaded "
                         "there.")
parser.add_argument('--save_path', default='../results/lenet-1/temp', type=str,
                    help="File path prefix to use to save the model and results")
parser.add_argument('--num_filters', default=8, type=int,
                    help="Number of filters per layer. If 0, this defaults to the number from the original ConvNet.")
parser.add_argument('--batch_size', default=4096, type=int,
                    help='Batch size')
parser.add_argument('--max_iter', default=100, type=int,
                    help="Maximum number of iterations to perform during the supervised training")
parser.add_argument('--eval_test_every', default=10, type=int,
                    help="Evaluate the performance on the test set every x iterations")
parser.add_argument('--loss', default='mnl', type=str,
                    help="Loss function to use. Either 'mnl' or 'square'")
parser.add_argument('--hessian_reg', default=0.03125, type=float,
                    help="Regularization on the hessian for the ULR method")
parser.add_argument('--update_lr_every', default=100, type=int,
                    help="Update the learning rate every x iterations")
parser.add_argument('--gpu', default='0', type=str,
                    help="Which GPU to use")

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Create the data loaders
train_loader, valid_loader, train_valid_loader, test_loader = mnist.get_dataloaders(batch_size=args.batch_size, data_path=args.data_path)

# Load and parse the parameters of the network
params = parse_config.load_config('../cfg/lenet-1.cfg')
if args.num_filters > 0:
    params['num_filters'] = [args.num_filters]*len(params['num_filters'])
layers = parse_config.create_layers(params)

# Create the network and initialize the filters using unsupervised training with spherical k-means
model = net.CKN(layers).to(defaults.device)
model.init(train_valid_loader)

# For 128 filters/layer we use a diagonal approximation to the Hessian due to memory constraints
if args.num_filters == 128:
    diag_hessian = True
else:
    diag_hessian = False

# Perform the supervised training
# For LeNet-1, layers 1 and 3 use the linear kernel and so their filters aren't trained.
train = train_supervised.TrainSupervised(train_loader, valid_loader, train_valid_loader, test_loader, model,
                                         nclasses=10, loss_name=args.loss.lower(), train_w_layers=[0, 2],
                                         diag_hessian=diag_hessian, update_lr_freq=args.update_lr_every,
                                         tau=args.hessian_reg, maxiter_outer=args.max_iter, save_path=args.save_path,
                                         eval_test_every=args.eval_test_every)
train.train()

# Plot the test accuracy vs. iteration
test_acc = pickle.load(open(args.save_path + '_results.pickle', 'rb'))['test_acc']
plt.plot(sorted(test_acc.keys()), [test_acc[key] for key in sorted(test_acc.keys())])
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy of LeNet-1 on MNIST with ' + str(args.num_filters) + ' filters per layer')
plt.show()
