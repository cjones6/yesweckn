import json
import numpy as np
import os
from torchvision import transforms
import torchvision.datasets as datasets

from . import create_data_loaders


def get_dataloaders(batch_size=128, data_path='../../data/ilsvrc2012_subset', num_workers=0):
    """
    Create data loaders for a subset of ImageNet.
    """
    train_dir = os.path.join(data_path, 'train')
    valid_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load all images into memory
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_images(train_dir, valid_dir,
                                                                                                   test_dir)

    train_dataset = create_data_loaders.PreloadedDataset(train_images, np.array(train_labels), train_transform)
    valid_dataset = create_data_loaders.PreloadedDataset(valid_images, np.array(valid_labels), valid_test_transform)
    test_dataset = create_data_loaders.PreloadedDataset(test_images, np.array(test_labels), valid_test_transform)

    return create_data_loaders.generate_dataloaders(train_dataset,
                                                    test_dataset,
                                                    valid_dataset=valid_dataset,
                                                    separate_valid_set=True,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)


def load_images(train_dir, valid_dir, test_dir):
    """
    Load the ImageNet images into memory using the list of images in the file imagenet_subset_file_list.json.
    """
    print('Loading images into memory... ', end='')
    files_list = json.load(open('../../misc/imagenet_subset_file_list.json', 'rb'))
    train_images = []
    train_labels = []
    for i, class_name in enumerate(sorted(files_list['train'].keys())):
        for image_file in files_list['train'][class_name]:
            train_images.append(datasets.folder.default_loader(os.path.join(train_dir, class_name, image_file)))
            train_labels.append(i)

    valid_images = []
    valid_labels = []
    for i, class_name in enumerate(sorted(files_list['val'].keys())):
        for image_file in files_list['val'][class_name]:
            valid_images.append(datasets.folder.default_loader(os.path.join(valid_dir, class_name, image_file)))
            valid_labels.append(i)

    test_images = []
    test_labels = []
    for i, class_name in enumerate(sorted(files_list['test'].keys())):
        for image_file in files_list['test'][class_name]:
            test_images.append(datasets.folder.default_loader(os.path.join(test_dir, class_name, image_file)))
            test_labels.append(i)
    print('done')

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels
