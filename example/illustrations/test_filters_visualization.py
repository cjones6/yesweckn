from matplotlib import pyplot as plt
from example.illustrations.filters import plot_filters, extract_filters


def visualize_filters(architecture, network_type, layer_idx, iteration=10000):
    filters = extract_filters(architecture, network_type, layer_idx, iteration)
    formatting = {'all-cnn': 'All-CNN', 'alexnet': 'AlexNet', 'lenet5': 'LeNet5', 'convnet': 'ConvNet', 'ckn': 'CKN'}
    fig = plot_filters(filters, 32)
    fig.suptitle(f'{formatting[architecture]} {formatting[network_type]} Layer {layer_idx} Iteration {iteration}', y=0.05)

    plt.show()
    return fig


if __name__ == '__main__':
    architectures = ['all-cnn', 'lenet5', 'alexnet']
    layers_idx_arch = {'all-cnn': [i for i in range(9)], 'lenet5': [0, 1, 2], 'alexnet': [0, 1, 2, 3, 4]}
    for architecture in architectures:
        layer_idxs = layers_idx_arch[architecture]
        for layer_idx in layer_idxs:
            for network_type in ['convnet', 'ckn']:
                for iteration in range(500, 10001, 500):
                    visualize_filters(architecture, network_type, layer_idx, iteration)