import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def get_optimizer(model:nn.Module, lr:float=None, weight_decay: float=None, 
                                                        optim_type: str=None) -> torch.optim.Optimizer:    
    ''' 
        optim_type (str): optimizer type {SGD, Adam, AdamW}
    '''

    if optim_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optim_type))