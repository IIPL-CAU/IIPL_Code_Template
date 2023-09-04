import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduer as lr_scheduler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_scheduler(optimizer:optim.Optimizer, scheduler_type:str=None, scheduler_params:dict=None, 
                  args:argparse.Namespace=None):
    ''' 입력으로 명시하지 않으면 args에서 가져온다.
    Args:
        scheduler_type (str): scheduler type {StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, 
                                ReduceLROnPlateau, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup}
        scheduler_parmas (dict): scheduler parameters (아래 명시된 params 필요)
    '''
    if scheduler_type is None:
        if args is None:
            raise ValueError("scheduler_type is None and args is None")
        else:
            scheduler_type = args.scheduler
    if scheduler_params is None:
        if args is None:
            raise ValueError("scheduler_params is None and args is None")
        else:
            scheduler_params = args.scheduler_params
    
    # lr_scheduler
    if scheduler_type == "StepLR":
        # scheduler_params = {"step_size":int, "gamma":float}
        return lr_scheduler.StepLR(optimizer, **scheduler_params)
    if scheduler_type == "MultiStepLR":
        # scheduler_params = {"milestones":list, "gamma":float}
        return lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    elif scheduler_type == "ExponentialLR":
        # scheduler_params = {"gamma":float}
        return lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == "CosineAnnealingLR":
        # scheduler_params = {"T_max":int, "eta_min":float}
        return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        # scheduler_params = {"mode":str, "factor":float, "patience":int, "verbose":bool}
        return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    
    # transformers
    elif scheduler_type == "get_cosine_schedule_with_warmup":
        # scheduler_params = {"num_warmup_steps":int, "num_training_steps":int}
        return get_cosine_schedule_with_warmup(optimizer, **scheduler_params)
    elif scheduler_type == "get_linear_schedule_with_warmup":
        # scheduler_params = {"num_warmup_steps":int, "num_training_steps":int}
        return get_linear_schedule_with_warmup(optimizer, **scheduler_params)
    else:
        raise ValueError("Unknown scheduler type: {}".format(scheduler_type))
    
