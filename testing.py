from models.model_init import model_init
import torch
from utils.data_load import data_load
import numpy as np
from sklearn.metrics import accuracy_score

def testing(args):
    if args.task == 'iris_classification':
        model = model_init(args)
        model.load_state_dict(torch.load(args.model_path))
        _, test_X, _, test_y = data_load(args)


        with torch.no_grad():
            y_pred = []
            for X in test_X:
                y_hat = model(X)
                y_pred.append(np.argmax(y_hat))
        
        print("Test Acc : " + str(accuracy_score(test_y, y_pred)))

    return None