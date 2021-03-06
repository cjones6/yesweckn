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

from src import default_params as defaults
from . import ckn_layer

if defaults.python3:
    import configparser
else:
    import ConfigParser as configparser


def load_config(path):
    """
    Load and parse the CKN configuration from the specified filepath.
    """
    default_param_dict = {'pad': '0',
                          'patch_size': '(3, 3)',
                          'stride': '[1, 1]',
                          'precomputed_patches': 'False',
                          'whiten': 'False',
                          'patch_kernel': 'rbf_sphere',
                          'filters_init': 'spherical-k-means',
                          'normalize': 'True',
                          'patch_sigma': '0.6',
                          'num_filters': '128',
                          'pool_kernel': 'average',
                          'pool_dim': '(1, 1)',
                          'subsample_factor': '(1, 1)',
                          'store_normalization': 'False',
                          'kww_reg': '0.001',
                          'num_newton_iters': '20',
                         }

    config = configparser.ConfigParser(default_param_dict)
    if defaults.python3:
        config.read_file(open(path))
    else:
        config.readfp(open(path))

    int_args = ['pad', 'num_filters', 'num_newton_iters']
    float_args = ['patch_sigma', 'kww_reg']
    str_args = ['patch_kernel', 'filters_init', 'pool_kernel']
    bool_args = ['precomputed_patches', 'whiten', 'normalize', 'store_normalization']
    list_int_args = ['patch_size', 'stride', 'pool_dim', 'subsample_factor']

    params = {}
    for arg_list, key_type in zip([int_args, float_args, str_args], [int, float, str]):
        for key in arg_list:
            params[key] = list(map(key_type, [config.get(section, key) for section in config.sections()]))

    for key in bool_args:
        values = [config.get(section, key) for section in config.sections()]
        params[key] = [values[i].lower() == 'true' for i in range(len(values))]

    for key in list_int_args:
        values = [config.get(section, key) for section in config.sections()]
        params[key] = [eval(values[i]) for i in range(len(values))]

    return params


def create_layers(params):
    """
    Create the layers of a CKN based on the input parameters.
    """
    n_layers = len(params['num_filters'])
    layers = []
    for layer_num in range(n_layers):
        layer = ckn_layer.CKNLayer(layer_num,
                                   params['patch_size'][layer_num],
                                   params['patch_kernel'][layer_num],
                                   params['num_filters'][layer_num],
                                   params['subsample_factor'][layer_num],
                                   padding=params['pad'][layer_num],
                                   stride=params['stride'][layer_num],
                                   precomputed_patches=params['precomputed_patches'][layer_num],
                                   whiten=params['whiten'][layer_num],
                                   filters_init=params['filters_init'][layer_num],
                                   normalize=params['normalize'][layer_num],
                                   patch_sigma=params['patch_sigma'][layer_num],
                                   pool_kernel=params['pool_kernel'][layer_num],
                                   pool_dim=params['pool_dim'][layer_num],
                                   store_normalization=params['store_normalization'][layer_num],
                                   kww_reg=params['kww_reg'][layer_num],
                                   num_newton_iters=params['num_newton_iters'][layer_num]
                                   )
        layers.append(layer)

    return layers
