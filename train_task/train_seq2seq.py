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
import wandb

from models.tokenizer.tokenizer_init import tokenizer_load
logger = get_logger("Training")



def train_seq2seq(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Load
    train_src_list, train_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                               seed=args.seed, mode='train')
    valid_src_list, valid_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                               seed=args.seed, mode='valid')
    args.num_classes = len(set(train_trg_list))
    logger.info(f'{args.dataset} data_load finish')

    # Model Load
    logger.info(f'start {args.model} model init')
    model = model_init(args)
    model.to(device)

    wandb.watch(model)
    logger.info(f'{args.model} model loaded')

    logger.info(f'{args.task} start train!')

    # tokenizer init
    src_tokenizer = tokenizer_load(args)
    trg_tokenizer = tokenizer_load(args)
    # Train datset setting
    custom_dataset_dict = dict()
    custom_dataset_dict['src_tokenizer'] = src_tokenizer
    custom_dataset_dict['trg_tokenizer'] = trg_tokenizer
    custom_dataset_dict['src_list'] = train_src_list['src_a']
    custom_dataset_dict['trg_list'] = train_trg_list['trg']
    train_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)

    # Valid dataset setting
    custom_dataset_dict['src_list'] = valid_src_list['src_a']
    custom_dataset_dict['trg_list'] = valid_trg_list['trg']
    valid_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)

    # Dataloader setting
    dataloader_dict = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=args.num_workers)
    }

    # Optimizer setting
    optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_dict['train']) * args.epochs)
    criterion = nn.CrossEntropyLoss()

    Best_loss = np.inf
    for epoch in range(args.epochs):
        val_loss = 0
        print(f"Epoch {epoch + 1}/{args.epochs}")

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                # write_log(logger, 'Validation start...')
                model.eval()
                logger.info(f"validation...")

            for batch in tqdm(dataloader_dict[phase]):

                # Optimizer gradient setting
                optimizer.zero_grad()

                # Input setting
                src_sequence = batch['src_sequence'].to(device)
                src_attention_mask = batch['src_attention_mask'].to(device)
                trg_sequence = batch['trg_sequence'].to(device)
                # trg_attention_mask = batch['trg_attention_mask'].to(device)

                # Model processing
                with torch.set_grad_enabled(phase == 'train'):
                    encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_attention_mask)
                    decoder_out = model.decode(trg_input_ids=trg_sequence, encoder_hidden_states=encoder_out, encoder_attention_mask=src_attention_mask)

                # Loss back-propagation
                decoder_out = decoder_out.view(-1, decoder_out.size(-1))
                trg_sequence = trg_sequence.view(-1)
                loss = criterion(decoder_out, trg_sequence)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                if phase == 'valid':
                    valid_loss += loss.item()

        val_loss /= len(dataloader_dict['valid'])
        if val_loss < Best_loss:
            torch.save(model.state_dict(), args.model_path)
            Best_loss = val_loss
        logger.info(f"validation finish! Loss : {val_loss} / best loss : {Best_loss}")
