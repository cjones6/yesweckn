from __future__ import division

import os
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.spatial import distance
from dppy.finite_dpps import FiniteDPP
import numpy as np
import sys


# from experiments.illustrations.interactive_visualization import visualize_filters
from src.opt.opt_structures import Model


def extract_all_filters_convnet(model, layer, architecture):
    if architecture == 'all-cnn':
        if layer == 0:
            filters = model.features[0].weight
            filters = filters.view(filters.shape[0], 3, 3, 3)
        else:
            assert model.features[layer]._get_name() == 'Conv2d'
            filters = model.features[layer].weight
    elif architecture in ['lenet5', 'alexnet']:
        assert model.layers[layer]._get_name() == 'Conv2d'
        filters = model.layers[layer].weight
    else:
        raise NotImplementedError
    return filters.cpu().detach().numpy()


def extract_all_filters_ckn(model, architecture, layer):
    if architecture == 'lenet5':
        layer = 2*layer
    if architecture == 'alexnet':
        model = model.module
    filters = model.layers[layer].W
    n_filters, patch_size = model.layers[layer].n_filters, model.layers[layer].patch_size
    n_channels = filters.shape[1] // (patch_size[0] * patch_size[1])
    filters = filters.view(filters.shape[0], n_channels, patch_size[0], patch_size[1])

    return filters.cpu().detach().numpy()


def sample_diverse_filters(filters, rng, bandwidth=0.01):
    vec_filters = filters.reshape(filters.shape[0], -1)
    similarity_matrix = np.exp(-distance.cdist(vec_filters, vec_filters)**2/bandwidth)
    DPP = FiniteDPP(kernel_type='likelihood',
                    projection=False,
                    **{'L': similarity_matrix})
    DPP.sample_exact(mode='GS', random_state=rng)
    return filters[DPP.list_of_samples[0]]


def plot_filters(filters, nfigs=None):
    if nfigs is None:
        nfigs = filters.shape[0]
    ncol = int(np.sqrt(nfigs) +1)
    nrow = int(np.ceil(nfigs / ncol))

    fig = plt.figure(figsize=(ncol, nrow))
    gs1 = gridspec.GridSpec(nrow, ncol)
    gs1.update(wspace=0.05, hspace=0.05)

    min_val = max(-1, np.min(filters))
    max_val = min(1, np.max(filters))

    count = 0
    for j in range(0, ncol):
        for i in range(0, nrow):
            if i * ncol + j < filters.shape[0]:
                ax = plt.subplot(gs1[count])
                im = ax.imshow(filters[i * ncol + j, :], cmap='Greens',
                               vmin=min_val, vmax=max_val, aspect='auto')
                ax.axis('off')
                count += 1
    return fig


def list_conv_layers(model, architecture):
    if architecture == 'all-cnn':
        conv_layers = [0]
        for i, layer in enumerate(model.features):
            if layer._get_name() == 'Conv2d':
                conv_layers.append(i)
    elif architecture in ['lenet5', 'alexnet']:
        conv_layers = []
        for i, layer in enumerate(model.layers):
            if layer._get_name() == 'Conv2d':
                conv_layers.append(i)
    else:
        raise NotImplementedError

    return conv_layers


def extract_filters(architecture, network_type, layer_idx, iteration):
    path_filters = f'filters/{architecture}_{network_type}_{layer_idx}_{iteration}.pickle'
    print(f'{architecture}_{network_type}_{layer_idx}_{iteration}')
    if not os.path.exists(path_filters):
        seed = 0
        rng = np.random.RandomState(seed)

        if architecture == 'all-cnn':
            if network_type == 'ckn':
                path_model = 'trained_nets/cifar10_all-cnn-c_ckn' \
                            f'/0.6_0.03125_32_20_ulr-sgo_0_0.015625_1644018388.4159172_model_{iteration}.pickle'
            else:
                path_model = 'trained_nets/cifar10_all-cnn-c_convnet' \
                            f'/0.0625_0.001953125_32_0_0.25_1644018343.2477334_model_{iteration}.pickle'
        elif architecture == 'lenet5':
            if network_type == 'ckn':
                path_model = 'trained_nets/mnist_lenet-5_ckn' \
                            f'/0.6_0.0078125_32_ulr-sgo_0_0.03125_1648852147.5136604_model_{iteration}.pickle'
            else:
                path_model = 'trained_nets/mnist_lenet-5_convnet' \
                            f'/0.0625_0_0.0078125_None_0_0.125_1648842900.652788_model_{iteration}.pickle'
        elif architecture == 'alexnet':
            if network_type == 'ckn':
                path_model = 'trained_nets/imagenet_subset_alexnet_ckn' \
                            f'/0.6_0.03125_rbf_sphere_1.5_32_0_0.015625_1649265499.746535_model_{iteration}.pickle'
            else:
                path_model = 'trained_nets/imagenet_subset_alexnet_convnet' \
                            f'/0.03125_0.001953125_32_0_0.25_1649181333.5809035_model_{iteration}.pickle'
        else:
            raise NotImplementedError
        model = Model()
        # If on a cpu, change model_dict = torch.load(path) in src.opt.opt_structurrs.Model.load to
        # model_dict = torch.load(path, map_location='cpu')
        model.load(path_model)
        model = model.model
        if network_type == 'ckn':
            filters = extract_all_filters_ckn(model, architecture, layer_idx)
        else:
            conv_layers = list_conv_layers(model, architecture)
            filters = extract_all_filters_convnet(model, conv_layers[layer_idx], architecture)
        filters = filters.reshape(-1, filters.shape[-2], filters.shape[-1])
        sampled_filters = sample_diverse_filters(filters, rng)
        if len(sampled_filters) > 4:
            filters = sampled_filters
        os.makedirs(os.path.dirname(path_filters), exist_ok=True)
        pickle.dump(filters, open(path_filters, 'wb'))
    else:
        filters = pickle.load(open(path_filters, 'rb'))
    return filters
