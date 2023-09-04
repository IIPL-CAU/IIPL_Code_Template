from models.model_init import model_init
from utils.data_load import data_load
from utils.optimizer import get_optimizer
from utils.scheduler import get_linear_schedule_with_warmup
from preprocessing import _tokenizer
from torch import nn

#tokenizer 적용 이전 잠시 berttokenizer 사용
from transformers import BertTokenizer

import torch


def training(args):
    
    model = model_init(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if args.task == 'iris_classification':
        losses = []
        X_train, _, y_train, _ = data_load(args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for i in range(args.epochs):
            model.train()
            y_pred = model(X_train)
            
            loss = criterion(y_pred, y_train)
            losses.append(loss)
            if i % 10 ==0:
                print(f'epoch {i}, loss is {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), args.model_path)
    
    if args.task =='single_text_classification':
        if args.model == "bert-base-uncased":

            #tokenizer init함수 필요
            tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
            #data_load
            """
            total_src_list, total_trg_list = data_load(args)

            train_src_list = total_src_list['train']
            valid_src_list = total_src_list['valid']
            train_trg_list = total_trg_list['train']
            valid_trg_list = total_trg_list['valid']

            #train_dataset = Custom_Dataset(tokenizer, train_src_list, train_trg_list, args)
            #val_dataset = Custom_Dataset(tokenizer, valid_src_list, valid_trg_list, args)
            #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
            """
            
            
            optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay, optim_type=args.optim_type)
            #total_steps = len(train_dataloader) * args.epochs
            total_steps = 100 # 테스트용입니다.
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            print(scheduler)

            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['src_sequence'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                if i % 10 ==0:
                    #logger 적용 필요
                    print(f'epoch {i}, loss is {loss}')
                
                #test 코드에 metric 작성되면 validation코드도 추가
                

        torch.save(model.state_dict(), args.model_path)





    return None