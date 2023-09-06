# Import modules

# Import PyTorch
import torch
# Import custom modules
from models.model_init import model_init
from utils.data_load import data_load
from utils.optimizer import get_optimizer
from utils.scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from preprocessing import _tokenizer
from torch import nn
from tqdm import tqdm

from preprocessing import tokenizer_load

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Load
    model = model_init(args)
    model.to(device)

    # Data Load
    total_src_list, total_trg_list = data_load(args)

    train_src_list = total_src_list['train']
    valid_src_list = total_src_list['valid']
    train_trg_list = total_trg_list['train']
    valid_trg_list = total_trg_list['valid']
    
    # if args.task == 'iris_classification':
    #     losses = []
    #     X_train, _, y_train, _ = data_load(args)
    #     criterion = torch.nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #     for i in range(args.epochs):
    #         model.train()
    #         y_pred = model(X_train)
            
    #         loss = criterion(y_pred, y_train)
    #         losses.append(loss)
    #         if i % 10 ==0:
    #             print(f'epoch {i}, loss is {loss}')
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     torch.save(model.state_dict(), args.model_path)
    
    if args.task =='single_text_classification':
        if args.model == "bert-base-uncased":

            #tokenizer init함수 필요
            tokenizer = tokenizer_load(args)
            # tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

            train_dataset = CustomDataset(tokenizer, train_src_list, train_trg_list)
            val_dataset = CustomDataset(tokenizer, valid_src_list, valid_trg_list)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
            
            optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.epochs)

            model.train()
            idx = 0 
            for epoch in range(args.epochs):
                print(f"Epoch {epoch + 1}/{args.epochs}")
                for batch in tqdm(train_dataloader):

                    # Optimizer gradient setting
                    optimizer.zero_grad()

                    # Input setting
                    input_ids = batch['src_sequence'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    # Model processing
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    # Loss back-propagation
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                print(f'Epoch {epoch + 1}/ loss : {loss}')
                #test 코드에 metric 작성되면 validation코드도 추가        
        torch.save(model.state_dict(), args.model_path)

    if args.task =='multi_text_classification':
        pass

    if args.task =='machine_translation':
        pass

    return None