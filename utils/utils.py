import os
import torch
import numpy as np
import logging
import argparse

import wandb
import random
import string

# seed setting
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# logger setting
def get_logger(logger_path:str=None, logger_name:str=None, args:argparse.Namespace=None):
    ''' 
    Args:
        logger_path (str): logger saving path (.log file)
        logger_name (str): logger name

    logger = get_logger(logger_path, logger_name, args)
    logger.info("message")
    '''
    if logger_path is None:
        if args is None:
            raise ValueError("logger_path is None and args is None")
        else:
            logger_path = args.logger_path
    if logger_name is None:
        if args is None:
            raise ValueError("logger_name is None and args is None")
        else:
            logger_name = args.logger_name
    # logger file generation        
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# wandb setting
def init_wandb(project_name="project", run_name:str=None, args:argparse.Namespace=None):
    '''
    로그인하고 필요한 부분에 적절히 적용하면 됨.
    $ pip install wandb
    $ wandb login

    # init wandb
    wandb.init(project="project_name", name="run_name", config=args)
    # hyperparameter logging
    wandb.config.update(args) 
    # metric logging (dict) 다양한 형태로 logging 가능
    wandb.log({"loss": loss}, step=epoch) 
    # alert 가능
    wandb.alert(title="alert title", text="alert text")
    '''
    if run_name is None:
        if args is None:
            run_name = get_run_name() # random name generation
        else:
            run_name = args.run_name

    wandb.init(project=project_name, name=run_name,  config=args.__dict__)

# size 만큼의 랜덤한 문자열 생성
def get_run_name(size=12, args:argparse.Namespace=None):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(size))

##??
def list_str2float(data):
    ret = []
    for s in data:
        ret.append(float(s))
    
    return ret