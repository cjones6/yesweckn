import torch.utils.data
import random
import torch

dataloader_timeout = 0


def generate_dataloaders(train_dataset, test_dataset, valid_dataset=None, separate_valid_set=False, valid_size=0,
                         batch_size=128, num_workers=0):
    """
    Create data loaders given the corresponding datasets from the PyTorch Dataset class.
    """
    if separate_valid_set:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=dataloader_timeout)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=dataloader_timeout)
    elif valid_size > 0:
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        train_novalid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[valid_size:])
        valid_dataset = torch.utils.data.dataset.Subset(train_dataset, indices[:valid_size])

        train_loader = torch.utils.data.DataLoader(train_novalid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=dataloader_timeout)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=dataloader_timeout)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=False,
                                                   timeout=dataloader_timeout)
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, drop_last=False, pin_memory=True,
                                              timeout=dataloader_timeout)

    return train_loader, valid_loader, test_loader


class PreloadedDataset(torch.utils.data.Dataset):
    """
    Dataset class for data that has already been loaded into memory that allows for transformations.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
