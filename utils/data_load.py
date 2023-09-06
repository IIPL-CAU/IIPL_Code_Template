import os
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
import csv
from sklearn.model_selection import train_test_split
# from utils.utils import list_str2float


# data loader by yeonghwa
def data_load(dataset_path:str=None, data_split_ratio:list=None, seed:int=None, 
                                                                    mode:str="train") -> (list, list):
    ''' 
        dataset_path (str): dataset name {imdb, ...} or dataset path
        data_split_ratio (list or None): [train, validation, test] split ratio eg., [0.8, 0.1, 0.1]
                                        *** if None, using original dataset split ratio ***
        mode (str): {train, valid, test}
    '''
            
    # huggingface dataset인 경우
    if not dataset_path.endswith(".csv") or dataset_path.endswith(".tsv") or dataset_path.endswith(".json"):
        # IMDB
        if dataset_path == "imdb" or dataset_path == "IMDB":
            if data_split_ratio is None:
                raise ValueError("IMDB only have 'train', 'test', 'unsupervised'. We need 'data_split_ratio'")

            # type : {train, test, unsupervised}
            raw_dataset = load_dataset("imdb") # <class 'datasets.arrow_dataset.Dataset'>
            combined_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])
            combined_dataset = combined_dataset.to_dict() # dict_keys(['text', 'label'])
            
            src_list, trg_list = split_data(dataset=combined_dataset, ratio=data_split_ratio, seed=seed, mode=mode)

            return src_list, trg_list # dict, dict
        # SST2
        elif dataset_path == "sst2" or dataset_path == "SST2":
            peint("SST2")
            
    # local file인 경우
    else:
        return [], []

# split data to train, validation, test
def split_data(dataset:dict, ratio:list, seed:int, mode:str) -> (list, list):
    src_key = list(dataset.keys())[0]
    trg_key = list(dataset.keys())[1]

    tr_rest_ratio = ratio[1]+ratio[2]
    val_test_ratio = ratio[2] / (ratio[1] + ratio[2])

    tr_src_list, rest_src_list, tr_trg_list, rest_trg_list = train_test_split(dataset[src_key], dataset[trg_key], 
                                                                random_state=seed,  test_size=tr_rest_ratio)
    if mode == "train":
        return tr_src_list, tr_trg_list
    
    # "valid" or "test"
    val_src_list, test_src_list, val_trg_list, test_trg_list = train_test_split(rest_src_list, rest_trg_list, 
                                                                random_state=seed, test_size=val_test_ratio)
    if mode == "valid":
        return val_src_list, val_trg_list
    elif mode == "test":
        return test_src_list, test_trg_list

        
# from local
# def data_load(args):
#     if args.file_type == 'csv':
#         if args.dataset == 'iris':
#             #return shape => X : n x 4, y : n x 1  
#             X = []
#             y = []
#             with open(args.file_dir, 'r', encoding='utf-8') as f :
#                 rdr = csv.reader(f)
#                 for line in rdr:
#                     try:
#                         X.append(list_str2float(line[1:5]))
#                     except:
#                         continue
#                     if line[5] == 'Iris-setosa':
#                         y.append(0)
#                     elif line[5] == 'Iris-Iris-versicolor': 
#                         y.append(1)
#                     else : y.append(2)
#                 f.close()
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             X_train = torch.FloatTensor(X_train)
#             X_test = torch.FloatTensor(X_test)
#             y_train = torch.LongTensor(y_train)
#             y_test = torch.LongTensor(y_test)

#             return X_train, X_test, y_train, y_test
        
#         if args.dataset == 'IMDB':
#             train_file_name = args.file_dir + "/train.csv"
#             test_file_name = args.file_dir + "/test.csv"
#             tr_src_list = []
#             tr_trg_list = []
#             val_src_list = []
#             val_trg_list = []
#             test_src_list = []
#             test_trg_list = []
#             label_map = {'positive' : 1, 'negative' : 0}
#             with open(train_file_name, 'r', encoding='utf-8') as f :
#                 rdr = csv.reader(f)
#                 for line in rdr:
#                     if line[0] == 'comment':
#                         continue
#                     else :
#                         tr_src_list.append(line[0])
#                         tr_trg_list.append(label_map[line[1]])
            
#             with open(test_file_name, 'r', encoding='utf-8') as f :
#                 rdr = csv.reader(f)
#                 for line in rdr:
#                     if line[0] == 'comment':
#                         continue
#                     else :
#                         test_src_list.append(line[0])
#                         test_trg_list.append(label_map[line[1]])
            
#             tr_src_list, val_src_list, tr_trg_list, val_trg_list = train_test_split(tr_src_list, tr_trg_list, test_size=args.val_ratio, random_state=42, shuffle=False)
#             assert len(tr_src_list) == len(tr_trg_list) and len(val_src_list) == len(val_trg_list) and len(test_src_list) == len(test_trg_list)

#             total_src_list = {'train' : tr_src_list, 'valid' : val_src_list, 'test' : test_src_list}
#             total_trg_list = {'train' : tr_trg_list, 'valid' : val_trg_list, 'test' : test_trg_list}
#             print("Train Dataset : " +str(len(tr_src_list)))
#             print("valid Dataset : " +str(len(val_src_list)))
#             print("test Dataset : " +str(len(test_src_list)))

#             return total_src_list, total_trg_list
            

            
                    

# if __name__ == '__main__':
#     a, b = data_load(dataset_path="imdb", seed=42, data_split_ratio=[0.8, 0.1, 0.1])
#     print(len(a))