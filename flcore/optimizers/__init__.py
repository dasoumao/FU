import torch

def get_optimizer(optimizer, param, **kwargs):
    optim = None
    if optimizer == "sgd":
        optim = torch.optim.SGD(param, **kwargs)
    elif optimizer == "adam":
        optim = torch.optim.Adam(param, **kwargs)
    else:
        raise NotImplementedError
    return optim

    