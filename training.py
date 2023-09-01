from models.model_init import model_init
from utils.data_load import data_load
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

    return None