import torch
import torchvision

from . import create_data_loaders


def get_dataloaders(valid_size=10000, batch_size=128, data_path='../data/MNIST', num_workers=0):
    """
    Create data loaders for MNIST.
    """
    def std(x):
        return (x.type(torch.get_default_dtype())/255.0-0.1307)/0.3081

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=std)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=std)

    train_dataset = create_data_loaders.PreloadedDataset(train_dataset.data.unsqueeze(1), train_dataset.targets.numpy(),
                                                         std)
    test_dataset = create_data_loaders.PreloadedDataset(test_dataset.data.unsqueeze(1), test_dataset.targets.numpy(),
                                                        std)

    return create_data_loaders.generate_dataloaders(train_dataset, test_dataset, separate_valid_set=False,
                                                    valid_size=valid_size, batch_size=batch_size,
                                                    num_workers=num_workers)
