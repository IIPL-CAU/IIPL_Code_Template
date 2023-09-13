from models.model_init import model_init
import torch
from torch.utils.data import DataLoader
from utils.data_load import data_load
from models.dataset.dataset_init import dataset_init
import numpy as np
import wandb

from tqdm import tqdm
from utils import metric
from models.tokenizer.tokenizer_init import tokenizer_load

def test_seq2seq(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")