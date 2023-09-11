# Import modules

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

from models.tokenizer.tokenizer_init import tokenizer_load
logger = get_logger("Training")


def training(args):
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
    logger.info(f'{args.model} model loaded')

    logger.info(f'{args.task} start train!')
    if args.task =='single_text_classification':
        if args.model == "bert-base-uncased":

            # tokenizer init
            src_tokenizer = tokenizer_load(args)

            # Train dataset setting
            custom_dataset_dict = dict()
            custom_dataset_dict['src_tokenizer'] = src_tokenizer
            custom_dataset_dict['src_list'] = train_src_list
            custom_dataset_dict['trg_list'] = train_trg_list
            train_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          pin_memory=True, num_workers=args.num_workers)

            # Valid dataset setting
            custom_dataset_dict['src_list'] = valid_src_list
            custom_dataset_dict['trg_list'] = valid_trg_list
            valid_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                          pin_memory=True, num_workers=args.num_workers)
            
            # Optimizer setting
            optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs)
            criterion = nn.CrossEntropyLoss()

            Best_acc = 0
            for epoch in range(args.epochs):
                model.train()
                logger.info(f"Epoch {epoch + 1}/{args.epochs}")
                for batch in tqdm(train_dataloader):

                    # Optimizer gradient setting
                    optimizer.zero_grad()

                    # Input setting
                    src_sequence = batch['src_sequence'].to(device)
                    src_attention_mask = batch['src_attention_mask'].to(device)
                    trg_label = batch['trg_label'].to(device)

                    # Model processing
                    outputs = model(input_ids=src_sequence, attention_mask=src_attention_mask)

                    # Loss back-propagation
                    loss = criterion(outputs, trg_label)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                logger.info(f'Epoch {epoch + 1}/ loss : {loss}')
                logger.info(f"validation start!")
                total_predictions = []
                total_labels = []
                for batch in tqdm(valid_dataloader):
                    # Input setting
                    src_sequence = batch['src_sequence'].to(device)
                    src_attention_mask = batch['src_attention_mask'].to(device)
                    trg_label = batch['trg_label'].to(device)
                    
                    with torch.no_grad(): 
                        outputs = model(input_ids=src_sequence, attention_mask=src_attention_mask)

                        logits = outputs.detach().cpu().numpy() 
                        predictions = np.argmax(logits, axis=1)
                        labels = trg_label.detach().cpu().numpy() 

                        total_predictions.extend(predictions)
                        total_labels.extend(labels)
                val_acc = metric.accuracy_score(total_labels,total_predictions)
                if val_acc > Best_acc : 
                    torch.save(model.state_dict(), args.model_path)
                    Best_acc = val_acc
                logger.info(f"validation finish! Acc : {val_acc} / best acc : {Best_acc}")


    if args.task =='multi_text_classification':
        pass

    if args.task =='machine_translation':
        pass

    return None