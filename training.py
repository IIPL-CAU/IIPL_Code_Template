from models.model_init import model_init
from utils.data_load import data_load
from utils.utils import optimizer_init
import torch

def training(args):
    
    model = model_init(args)
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
    
    if args.task =='bert-base-uncased-classification':
        losses = []
        # train_loader 불러오기 data_load -> train, validation, test 데이터 로더로 예상했습니다.
        train_loader, _, _ = data_load(args)
        # optimizer의 경우 args.optim을 추가하여 진행할 필요가 있습니다. 우선 Adam으로 진행 부탁드립니다.
        optimizer = optimizer_init(args)
        
        for encoded, label in train_loader:
            optimizer.zero_grad()
            outputs = model(**encoded, labels=label)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            if i % 10 ==0:
                print(f'epoch {i}, loss is {loss}')

        torch.save(model.state_dict(), args.model_path)





    return None