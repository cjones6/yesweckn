# coding=utf-8
import configparser

from src.model.ckn import ckn_layer


def load_config(path):
    """
    Load and parse the CKN configuration from the specified filepath.
    """
    # Default parameters for a single layer
    default_param_dict = {'pad': '0',  # Amount of zero-padding for each side of the input
                          'patch_size': '(3, 3)',  # Dimensions of the patches
                          'stride': '[1, 1]',  # Stride with which to extract patches
                          'precomputed_patches': 'False',  # Whether the inputs consist of precomputed patches
                          'whiten': 'False',  # Whether to whiten the input patches
                          'patch_kernel': 'rbf_sphere',  # Kernel to use in the Nyström approximation
                          'filters_init': 'spherical-k-means',  # How to initialize the filters
                          'normalize': 'True',  # Whether to normalize the patches so they have norm 1
                          'patch_sigma': '0.6',  # Bandwidth of the kernel
                          'matern_order': '1.5',  # Order of the Matérn kernel (if applicable)
                          'num_filters': '128',  # Number of filters to use in the Nyström approximation
                          'pool_kernel': 'average',  # Kernel to use for pooling
                          'pool_dim': '(1, 1)',  # Dimensions of the pooling
                          'subsample_factor': '(1, 1)',  # Subsampling factor (applied after pooling)
                          'store_normalization': 'False',  # Whether to store (k(WW^T)+kww_reg I)^{-1/2}
                          'kww_reg': '0.001',  # Factor multiplied by the identity matrix to be added to k(WW^T)
                          'num_newton_iters': '20',  # Number of iterations of the intertwined Newton method to perform
                          }

    config = configparser.ConfigParser(default_param_dict)
    config.readfp(open(path))

    int_args = ['pad', 'num_filters', 'num_newton_iters']
    float_args = ['patch_sigma', 'matern_order', 'kww_reg']
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
                                   matern_order=params['matern_order'][layer_num],
                                   pool_kernel=params['pool_kernel'][layer_num],
                                   pool_dim=params['pool_dim'][layer_num],
                                   store_normalization=params['store_normalization'][layer_num],
                                   kww_reg=params['kww_reg'][layer_num],
                                   num_newton_iters=params['num_newton_iters'][layer_num],
                                   )
        layers.append(layer)

    return layers
