import os
import sys
import torch
import numpy as np
import logging
import argparse
import wandb
import random
import string

# seed setting
def set_seed(seed:int=None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# logger setting
def get_logger(logger_name:str=None):
    '''
        logger = get_logger(logger_path, logger_name, args)
        logger.info("message")
    '''
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        #handler.setFomatter(logging.Formatter("[%(asctime)s] %(message)s"))

        logger.addHandler(handler)
    
    #https://docs.python.org/3/library/logging.handlers.html
    #check the library(link above) but there is no StreamHandler.setFomatter : maybe library modified? 
    
    return logger

# wandb setting
def init_wandb(wandb_dir="./wandb", project_name="project", run_name:str=None, args:argparse.Namespace=None):
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
        run_name = get_run_name() # random name generation

    wandb.init(dir=wandb_dir, project=project_name, name=run_name,  config=args.__dict__)

# size 만큼의 랜덤한 문자열 생성
def get_run_name(size=12, args:argparse.Namespace=None):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(size))
