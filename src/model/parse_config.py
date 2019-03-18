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
from collections import OrderedDict
import ConfigParser

import ckn_layer


def load_config(path):
    """
    Load and parse the CKN configuration from the specified filepath.
    """
    setup_cfg = ConfigParser.ConfigParser(dict_type=MultiOrderedDict)
    setup_cfg.readfp(open(path))
    nl = len(setup_cfg.get('convolutional', 'num_filters'))

    config = ConfigParser.ConfigParser({'pad': ['0']*nl,
                                        'patch_size': [(3, 3)] * nl,
                                        'stride': [[1, 1] * nl],
                                        'precomputed_patches': ['False']*nl,
                                        'whiten': ['False'] * nl,
                                        'patch_kernel': ['rbf_sphere'] * nl,
                                        'filters_init': ['spherical-k-means'] * nl,
                                        'normalize': ['True'] * nl,
                                        'patch_sigma': ['0.6'] * nl,
                                        'num_filters': ['128'] * nl,
                                        'pool_kernel': ['average']*nl,
                                        'pool_dim': [(1, 1)] * nl,
                                        'subsample_factor': [(1, 1)]*nl,
                                        'store_normalization': ['False']*nl,
                                        'kww_reg': ['0.001']*nl,
                                        'num_newton_iters': ['20']*nl,
                                        },
                                       dict_type=MultiOrderedDict)
    config.readfp(open(path))

    int_args = ['pad', 'num_filters', 'num_newton_iters']
    float_args = ['patch_sigma', 'kww_reg']
    str_args = ['patch_kernel', 'filters_init', 'pool_kernel']
    bool_args = ['precomputed_patches', 'whiten', 'normalize', 'store_normalization']
    list_int_args = ['patch_size', 'stride', 'pool_dim', 'subsample_factor']

    params = {}
    for arg_list, key_type in zip([int_args, float_args, str_args], [int, float, str]):
        for key in arg_list:
            params[key] = map(key_type, config.get('convolutional', key))

    for key in bool_args:
        values = config.get('convolutional', key)
        params[key] = [values[i].lower() == 'true' for i in range(nl)]

    for key in list_int_args:
        values = config.get('convolutional', key)
        params[key] = [eval(values[i]) for i in range(nl)]

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


class MultiOrderedDict(OrderedDict):
    """
    Class allowing an OrderedDict to have multiple keys with the same value.
    https://stackoverflow.com/questions/15848674/how-to-configparse-a-file-keeping-multiple-values-for-identical-keys
    """
    def __setitem__(self, key, value):
        if isinstance(value, list) and key in self:
            self[key].extend(value)
        else:
            super(OrderedDict, self).__setitem__(key, value)