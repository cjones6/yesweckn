import sys
import os
import streamlit as st
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle

# sys.path.append('../..')
# from example.illustrations.filters import plot_filters, extract_filters

architecture = st.sidebar.selectbox('Architecture', ['LeNet5 on MNIST', 'All-CNN on CIFAR10', 'AlexNet on ImageNet'])
streamlit_to_code = {'LeNet5 on MNIST': 'lenet5', 'All-CNN on CIFAR10': 'all-cnn', 'AlexNet on ImageNet': 'alexnet'}
architecture = streamlit_to_code[architecture]
iteration = st.sidebar.slider('Iteration', 500, 10000, step=500)
if architecture == 'all-cnn':
    nb_layers = 9
elif architecture == 'lenet5':
    nb_layers = 3
elif architecture == 'alexnet':
    nb_layers = 5
else:
    nb_layers = 0

layer = st.sidebar.slider('Layer', 1, nb_layers, step=1)


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


def visualize_filters(architecture, network_type, layer_idx, iteration=10000):
    plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})
    dir_name = os.path.dirname(os.path.abspath(__file__))
    path_filters = os.path.join(dir_name, f'filters/{architecture}_{network_type}_{layer_idx}_{iteration}.pickle')
    filters = pickle.load(open(path_filters, 'rb'))
    # filters = extract_filters(architecture, network_type, layer_idx, iteration)
    fig = plot_filters(filters, 32)
    return fig


formatting = {'all-cnn': 'All-CNN', 'lenet5': 'LeNet5', 'alexnet': 'AlexNet', 'convnet': 'ConvNet', 'ckn': 'CKN'}

col1, col2 = st.columns(2)

fig1 = visualize_filters(architecture, 'convnet', layer-1, iteration)
fig2 = visualize_filters(architecture, 'ckn', layer-1, iteration)

col1.subheader(f'{formatting[architecture]} ConvNet Layer {layer}\n #### Iteration {iteration}')
col1.pyplot(fig1)
col2.subheader(f'{formatting[architecture]} CKN Layer {layer}\n #### Iteration {iteration}')
col2.pyplot(fig2)


