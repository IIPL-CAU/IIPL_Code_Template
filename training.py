# Import modules
import numpy as np
# Import PyTorch
import torch
# Import custom modules
from models.model_init import model_init
from utils.data_load import data_load
from utils.optimizer import get_optimizer
from utils.scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils.utils import get_logger
from utils import metric
from models.dataset.dataset_init import dataset_init
from torch import nn
from tqdm import tqdm
from train.train_seq2label import train_seq2label
from train.train_seq2seq import train_seq2seq
import wandb

from models.tokenizer.tokenizer_init import tokenizer_load
logger = get_logger("Training")


def training(args):
    
    if args.task =='single_text_classification':
        train_seq2label(args)
        
    if args.task =='machine_translation':
        train_seq2seq(args)
