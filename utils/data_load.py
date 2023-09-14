import os
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
import pandas as pd
from tqdm import tqdm 
import csv
from sklearn.model_selection import train_test_split
# from utils.utils import list_str2float


# data loader
# data_split_ratio가 None인 경우 huggingface dataset의 original dataset split ratio 사용
def data_load(dataset_path:str, huggingface_data:bool=True, data_split_ratio:list=[0.8, 0.1, 0.1], seed:int=42, 
              mode:str="train") -> (dict, dict):
    '''
        dataset_path (str): dataset name {imdb, ...} or dataset path
        huggingface_data (bool) : {True, False} whether to use the huggingface dataset (datasets > load_dataset library)
        data_split_ratio (list or None): [train, validation, test] split ratio or None

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
            dataset_name, task_name = "imdb", None
            key_list = ['text', None, 'label']

        # SST2
        elif dataset_path == "sst2": # type : {train, validation, test}
            dataset_name, task_name = "glue", "sst2"
            key_list = ['sentence', None, 'label']
        
        # multi30k (en-de)
        elif dataset_path == "bentrevett/multi30k" or dataset_path == "multi30k": # type : {train, validation, test}
            dataset_name, task_name = "bentrevett/multi30k", None
            key_list = ['en', None, 'de']
        
        # QNLI
        elif dataset_path == "qnli": # type : {train, validation, test}
            dataset_name, task_name = "glue", "qnli"
            key_list = ['question', 'sentence', 'label']
        
        else: 
            raise ValueError("dataset_path must be specified (imdb, sst2, multi30k, qnli)")
        
        dataset = split_dataset(dataset_name=dataset_name, task_name=task_name, key_list=key_list,
                                                            ratio=data_split_ratio, seed=seed, mode=mode)

        src_dict = {'src_a' : dataset[key_list[0]], 
                    'src_b' : dataset[key_list[1]] if key_list[1] is not None else None}
        trg_dict = {'trg' : dataset[key_list[-1]]}

        return src_dict, trg_dict # dict, dict
    
    elif huggingface_data == False:
        if datapath.endswith('.tsv'):
            
        else:
            raise ValueError("dataset_path must be end with .tsv")
    else:
        raise ValueError("huggingface_data must be specified (True of False)")

# split data to train, validation, test
def split_dataset(dataset_name:str, task_name:str, key_list:list, ratio:list, seed:int, mode:str):
    if ratio is None:
        raw_dataset = load_dataset(dataset_name, task_name, split=mode)
        return raw_dataset
        
    else:
        raw_dataset = load_dataset(dataset_name, task_name)

        tr_rest_ratio = ratio[1]+ratio[2]
        val_test_ratio = ratio[2] / (ratio[1] + ratio[2])

        for key in list(raw_dataset.keys()): # train, validation, test
            if not key in ['train', 'validation', 'valid', 'test']:
                raw_dataset.pop(key)


        combined_dataset = concatenate_datasets([raw_dataset[split] for split in list(raw_dataset.keys())])
        tr_dataset, rest_dataset = train_test_split(combined_dataset, random_state=seed,  test_size=tr_rest_ratio)
        if mode == "train":
            return tr_dataset
        else:
            val_dataset, test_dataset = train_test_split(rest_dataset, random_state=seed, test_size=val_test_ratio)
            if mode == "valid":
                return val_dataset
            elif mode == "test":
                return test_dataset
        
def load_tsv(dataset_path:str, mode:str) -> (dict, dict):
    dataset_name = dataset_path.split('/')[-1].lower() # dataset name{imdb}
    dataset_path = os.path.join(dataset_path, f"{mode}.tsv")
    df = pd.read_csv(dataset_path, delimiter='\t', lineterminator='\n')
    
    if dataset_name == "imdb":
        key_list = ['text', None, 'label']
    elif dataset_name == 'qnli':
        key_list = ['question', 'sentence', 'label']
    else:
        raise ValueError("dataset_path must be end with name of dataset")
    
    src_dict = {'src_a' : df[key_list[0]].tolist(), 
                'src_b' : df[key_list[1]].tolist() if key_list[1] is not None else None}
    trg_dict = {'trg' : df[key_list[-1]].tolist()}

    return src_dict, trg_dict # dict, dict

if __name__ == '__main__':
    # a, b = data_load(dataset_path="imdb", seed=42, data_split_ratio=None)#[0.8, 0.1, 0.1])
    # a, b = data_load(dataset_path="bentrevett/multi30k", seed=42, data_split_ratio=None)#[0.8, 0.1, 0.1])
    # a, b = data_load(dataset_path="sst2", seed=42, data_split_ratio=None)#[0.8, 0.1, 0.1])
    # a, b = data_load(dataset_path="qnli", seed=42, data_split_ratio=None)#[0.8, 0.1, 0.1])
    print(a['src_a'][0])
    print(b['trg'][0])
    print(a['src_b'] is None)
