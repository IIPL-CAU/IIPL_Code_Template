import os
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm 
import csv
from sklearn.model_selection import train_test_split
# from utils.utils import list_str2float


# data loader 
# data_split_ratio가 None인 경우 huggingface dataset의 original dataset split ratio 사용
def data_load(dataset_path:str, huggingface_data:bool=True, data_split_ratio:list=[0.8, 0.1, 0.1], seed:int=42, 
              mode:str="train") -> (list, list):
    '''
        dataset_path (str): dataset name {imdb, ...} or dataset path
        huggingface_data = {True, False} whether to use the huggingface dataset (datasets > load_dataset library)
        data_split_ratio (list or None): [train, validation, test] split ratio 
                                        *** if None, using original dataset split ratio ***
        seed (int or None): random seed
        mode (str): {train, valid, test}
    '''
            
    # huggingface dataset인 경우
    if huggingface_data == True:
        dataset_path = dataset_path.lower()
        # IMDB
        if dataset_path == "imdb": # type : {train, test, unsupervised}

            if data_split_ratio is None:
                raise ValueError("data_split_ratio must be specified when using IMDB dataset")

            raw_dataset = load_dataset("imdb") # <class 'datasets.arrow_dataset.Dataset'>
            combined_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["test"]])
            combined_dataset = combined_dataset.to_dict() # dict_keys(['text', 'label'])
            
            src_list, trg_list = split_data(dataset=combined_dataset, ratio=data_split_ratio, seed=seed, mode=mode)
            
            src_dict = {'src_a' : src_list, 'src_b' : None}
            trg_dict = {'trg' : trg_list}
                        
            return src_dict, trg_dict # dict, dict
        
        # SST2
        elif dataset_path == "sst2": # type : {train, validation, test}
            # data_split_ratio가 None인 경우 original dataset split ratio 사용
            if data_split_ratio is None:

                if mode == "valid": mode = "validation"
                raw_dataset = load_dataset("glue", "sst2", split=mode)
                
                src_dict = {'src_a' : raw_dataset['sentence'], 'src_b' : None}
                trg_dict = {'trg' : raw_dataset['label']}
                
                return src_dict, trg_dict # dict, dict
            
            # data_split_ratio만큼 split
            else:
                raw_dataset = load_dataset("glue", "sst2")
                combined_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["validation"], raw_dataset["test"]])
                combined_dataset = combined_dataset.to_dict()

                src_list, trg_list = split_data(dataset=combined_dataset, ratio=data_split_ratio, seed=seed, mode=mode)
                
                src_dict = {'src_a' : src_list, 'src_b' : None}
                trg_dict = {'trg' : trg_list}
                        
                return src_dict, trg_dict # dict, dict
        
        # multi30k (en-de)
        elif dataset_path == "bentrevett/multi30k" or dataset_path == "multi30k": # type : {train, validation, test}
            # data_split_ratio가 None인 경우 original dataset split ratio 사용
            if data_split_ratio is None:

                if mode == "valid": mode = "validation"
                raw_dataset = load_dataset("bentrevett/multi30k", split=mode)
                
                src_dict = {'src_a' : raw_dataset['en'], 'src_b' : None}
                trg_dict = {'trg' : raw_dataset['de']}
                
                return src_dict, trg_dict # dict, dict

            # data_split_ratio만큼 split
            else: 
                raw_dataset = load_dataset("bentrevett/multi30k")
                combined_dataset = concatenate_datasets([raw_dataset["train"], raw_dataset["validation"], raw_dataset["test"]])
                combined_dataset = combined_dataset.to_dict() # dict_keys(['en', 'de'])

                src_list, trg_list = split_data(dataset=combined_dataset, ratio=data_split_ratio, seed=seed, mode=mode)
                
                src_dict = {'src_a' : src_list, 'src_b' : None}
                trg_dict = {'trg' : trg_list}
                
                return src_dict, trg_dict # dict, dict
            
    # local file인 경우
    elif huggingface_data == False:
        
        def _read_tsv(input_file, quotechar=None):
            with open(input_file, "r", encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                lines = []
                for line in reader:
                    lines.append(line)
                return lines
                 
        src_a_list = []
        src_b_list = []
        trg_list = []
        
        dataset_name = str(dataset_path).split('/')[-1]
        lines = _read_tsv(os.path.join(dataset_path, f"{mode}.tsv"))
        
        if dataset_name == 'imdb':   
            for (i, line) in enumerate(tqdm(lines)):
                if i == 0:
                    continue
                try:
                    src_a_list.append(line[0])
                    trg_list.append(line[1])
                except Exception as e:
                    continue
            src_dict = {'src_a' : src_a_list, 'src_b' : None}
            trg_dict = {'trg' : trg_list}
        
        elif dataset_name == 'qnli':   
            for (i, line) in enumerate(tqdm(lines)):
                if i == 0:
                    continue
                try:
                    src_a_list.append(line[0])
                    src_b_list.append(line[1])
                    trg_list.append(line[2])
                except Exception as e:
                    continue
            src_dict = {'src_a' : src_a_list, 'src_b' : src_b_list}
            trg_dict = {'trg' : trg_list}
         
        return src_dict, trg_dict # dict, dict

    else :
        raise ValueError("huggingface_data must be specified (True of False)")

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
    if mode == "valid" or mode == "validation":
        return val_src_list, val_trg_list

    elif mode == "test":
        return test_src_list, test_trg_list

# if __name__ == '__main__':
#     # a, b = data_load(dataset_path="imdb", seed=42, data_split_ratio=[0.8, 0.1, 0.1])
#     # data_load(dataset_path="bentrevett/multi30k", seed=42, data_split_ratio=[0.8, 0.1, 0.1])
#     a, b = data_load(dataset_path="sst2", seed=None, data_split_ratio=None, mode="train")
#     print(len(a))
#     print(a[0])
        
        
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
            

            
                    
