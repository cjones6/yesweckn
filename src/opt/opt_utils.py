import torch

from src import default_params as defaults


def get_batch(data):
    """
    Get a batch of training data (x, y) from the training set.
    """
    try:
        x, y = next(data.train_iter)
    except:
        data.train_iter = iter(data.train_loader)
        x, y = next(data.train_iter)
    x = x.type(torch.get_default_dtype()).to(defaults.device)
    y = y.to(defaults.device)

    return x, y


def compute_features(x, model, normalize=True, standardize=False, eps=1e-5):
    """
    Compute the features of the network on a mini-batch of inputs.
    """
    features = model(x.to(defaults.device))
    features = features.contiguous().view(features.shape[0], -1)

    if normalize:
        mean = torch.mean(features, 1)
        features = features - mean.unsqueeze(1)
        nrm = torch.mean(torch.norm(features, 2, 1))
        features = features / nrm
    if standardize:
        mean = torch.mean(features, 0, keepdim=True)
        sd = torch.clamp(torch.std(features, 0, keepdim=True), min=eps)
        features = (features - mean)/sd

    return features


def compute_all_features(train_loader, valid_loader, test_loader, model, normalize=True, standardize=False, eps=1e-5):
    """
    Generate features for all images in the training, validation, and test sets using the given model. Then normalize
    or standardize them.
    """
    with torch.autograd.set_grad_enabled(False):
        all_features = {'train': {'x': [], 'y': []}, 'valid': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}}
        for dataset_name, data_loader in zip(sorted(all_features.keys()), [test_loader, train_loader, valid_loader]):
            if data_loader is not None:
                for i, (x, y) in enumerate(data_loader):
                    x = x.type(torch.get_default_dtype()).to(defaults.device)
                    features = model(x)
                    features = features.contiguous().view(features.shape[0], -1).data.cpu()
                    all_features[dataset_name]['x'].append(features)
                    all_features[dataset_name]['y'].append(y)

                all_features[dataset_name]['x'] = torch.cat(all_features[dataset_name]['x'])
                all_features[dataset_name]['y'] = torch.cat(all_features[dataset_name]['y'])

        if normalize:
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].sub_(torch.mean(all_features[key]['x'], 1, keepdim=True))
            nrm = torch.mean(torch.norm(all_features['train']['x'], 2, 1))
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].div_(nrm)
        elif standardize:
            mean = torch.mean(all_features['train']['x'], 0, keepdim=True)
            sd = torch.clamp(torch.std(all_features['train']['x'], 0, keepdim=True), min=eps)
            for key in all_features.keys():
                if len(all_features[key]['x']) > 0:
                    all_features[key]['x'].sub_(mean)
                    all_features[key]['x'].div_(sd)

    return all_features['train']['x'], all_features['valid']['x'], all_features['test']['x'], \
           all_features['train']['y'], all_features['valid']['y'], all_features['test']['y'],


def compute_normalizations(model):
    """
    Compute the term k(WW^T)^{-1/2} for each layer of a CKN.
    """
    try:
        for layer_num in range(len(model.layers)):
            model.layers[layer_num].store_normalization = True
            with torch.autograd.set_grad_enabled(False):
                model.layers[layer_num].normalization = model.layers[layer_num].compute_normalization()
    except:
        for layer_num in range(len(model.module.layers)):
            model.module.layers[layer_num].store_normalization = True
            with torch.autograd.set_grad_enabled(False):
                model.module.layers[layer_num].normalization = model.module.layers[layer_num].compute_normalization()
    return model


def one_hot_embedding(y, n_dims):
    """
    Generate a one-hot representation of the input vector y.
    https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
    """
    y_tensor = y.data.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.shape[0], -1)

    return y_one_hot.type(torch.get_default_dtype()).to(defaults.device)


def print_results(iteration, results, header=False):
    """
    Print the results from one iteration.
    """
    if header:
        print('Iteration \t Train accuracy \t Valid accuracy \t Test accuracy \t Train loss \t Valid loss \t Test loss')
    print(iteration, '\t\t',
          '{:06.4f}'.format(results['train_accuracy']), '\t\t',
          '{:06.4f}'.format(results['valid_accuracy']), '\t\t',
          '{:06.4f}'.format(results['test_accuracy']), '\t',
          '{:06.4f}'.format(results['train_loss']), '\t',
          '{:06.4f}'.format(results['valid_loss']), '\t',
          '{:06.4f}'.format(results['test_loss']),
          )
