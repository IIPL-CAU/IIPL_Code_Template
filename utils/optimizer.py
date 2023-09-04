import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model:nn.Module, lr:float=None, weight_decay: float=None, 
                  optim_type: str=None, args: argparse.Namespace=None) -> torch.optim.Optimizer:    
    ''' 입력으로 명시하지 않으면 args에서 가져온다.
    Args:
        lr (float): learning rate
        weight_decay (float): weight decay
        optim_type (str): optimizer type {SGD, Adam, AdamW}
    '''
    if lr is None:
        if args is None:
            raise ValueError("lr is None and args is None")
        else: 
            lr = args.lr
    if weight_decay is None:
        if args is None:
            raise ValueError("weight_decay is None and args is None")
        else: 
            weight_decay = args.weight_decay
    if optim_type is None:
        if args is None:
            raise ValueError("optim_type is None and args is None")
        else: 
            optim_type = args.optim_type
    
    if optim_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optim_type))