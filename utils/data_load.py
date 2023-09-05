import csv
from sklearn.model_selection import train_test_split
import torch
from utils.utils import list_str2float


def data_load(args):
    if args.file_type == 'csv':
        if args.task == 'iris_classification':
            #return shape => X : n x 4, y : n x 1  
            X = []
            y = []
            with open(args.file_dir, 'r', encoding='utf-8') as f :
                rdr = csv.reader(f)
                for line in rdr:
                    try:
                        X.append(list_str2float(line[1:5]))
                    except:
                        continue
                    if line[5] == 'Iris-setosa':
                        y.append(0)
                    elif line[5] == 'Iris-Iris-versicolor': 
                        y.append(1)
                    else : y.append(2)
                f.close()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)

            return X_train, X_test, y_train, y_test