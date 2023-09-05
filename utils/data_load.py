import csv
from sklearn.model_selection import train_test_split
import torch
from utils.utils import list_str2float
import os

def data_load(args):
    if args.file_type == 'csv':
        if args.dataset == 'iris':
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
        
        if args.dataset == 'IMDB':
            train_file_name = args.file_dir + "/train.csv"
            test_file_name = args.file_dir + "/test.csv"
            tr_src_list = []
            tr_trg_list = []
            val_src_list = []
            val_trg_list = []
            test_src_list = []
            test_trg_list = []
            label_map = {'positive' : 1, 'negative' : 0}
            with open(train_file_name, 'r', encoding='utf-8') as f :
                rdr = csv.reader(f)
                for line in rdr:
                    if line[0] == 'comment':
                        continue
                    else :
                        tr_src_list.append(line[0])
                        tr_trg_list.append(label_map[line[1]])
            
            with open(test_file_name, 'r', encoding='utf-8') as f :
                rdr = csv.reader(f)
                for line in rdr:
                    if line[0] == 'comment':
                        continue
                    else :
                        test_src_list.append(line[0])
                        test_trg_list.append(label_map[line[1]])
            
            tr_src_list, val_src_list, tr_trg_list, val_trg_list = train_test_split(tr_src_list, tr_trg_list, test_size=args.val_ratio, random_state=42, shuffle=False)
            assert len(tr_src_list) == len(tr_trg_list) and len(val_src_list) == len(val_trg_list) and len(test_src_list) == len(test_trg_list)

            total_src_list = {'train' : tr_src_list, 'valid' : val_src_list, 'test' : test_src_list}
            total_trg_list = {'train' : tr_trg_list, 'valid' : val_trg_list, 'test' : test_trg_list}
            print("Train Dataset : " +str(len(tr_src_list)))
            print("valid Dataset : " +str(len(val_src_list)))
            print("test Dataset : " +str(len(test_src_list)))

            return total_src_list, total_trg_list
            

            
                    
