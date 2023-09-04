import os
import torch
import numpy as np
import logging
import argparse


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

##??
def list_str2float(data):
    ret = []
    for s in data:
        ret.append(float(s))
    
    return ret