# Import modules
import numpy as np
# Import PyTorch
import torch
# Import custom modules
from models.model_init import model_init
from utils.data_load import data_load
from torch.utils.data import DataLoader
from utils.utils import get_logger
from models.dataset.dataset_init import dataset_init
from torch import nn
from tqdm import tqdm
import wandb

from models.tokenizer.tokenizer_init import tokenizer_load
def test_seq2seq(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger("Training")
    
    # Data Load
    test_src_list, test_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                             seed=args.seed, mode='test')
    logger.info(f'{args.dataset_path} data_load finish')

    # Model Load
    logger.info(f'start {args.model} model init')
    model = model_init(args)
    model.to(device)

    logger.info(f'{args.model} model loaded')

    src_tokenizer = tokenizer_load(args)
    trg_tokenizer = tokenizer_load(args)

    # Train datset setting
    custom_dataset_dict = dict()
    custom_dataset_dict['src_tokenizer'] = src_tokenizer
    custom_dataset_dict['trg_tokenizer'] = trg_tokenizer
    custom_dataset_dict['src_list'] = test_src_list
    custom_dataset_dict['trg_list'] = test_trg_list
    test_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, 
                                 pin_memory=True, num_workers=args.num_workers)

    # Model Load
    # 짜야 함

    model.eval()
    hypothesis, references = list(), list()

    for batch in tqdm(test_dataloader):

        # Input setting
        src_sequence = batch['src_sequence'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        trg_sequence = batch['trg_sequence'].to(device)
        trg_attention_mask = batch['trg_attention_mask'].to(device)

        # Model processing
        with torch.no_grad():
            encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_attention_mask)

            # Decoding
            # decoding_dict = return_decoding_dict(args)
            predicted = model.generate(decoding_dict=decoding_dict, encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)

        if i == 0:
            print(trg_tokenizer.batch_decode(predicted)[0])
            print(trg_tokenizer.batch_decode(trg_sequence)[0])

        hypothesis.extend(trg_tokenizer.batch_decode(predicted, skip_special_tokens=True))
        references.extend(trg_tokenizer.batch_decode(trg_sequence, skip_special_tokens=True))
