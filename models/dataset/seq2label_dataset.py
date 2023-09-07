from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
import numpy as np

class Seq2Label_CustomDataset(Dataset):
    def __init__(self, src_tokenizer, src_list: list = list(), trg_list: list = list(), 
                 min_len: int = 10, src_max_len: int = 300):

        self.src_tensor_list = list()
        self.trg_tensor_list = list()
        
        self.src_tokenizer = src_tokenizer

        self.min_len = min_len
        self.src_max_len = src_max_len

        for src, trg in zip(src_list, trg_list):
            if min_len <= len(src):
                self.src_tensor_list.append(src)
            self.trg_tensor_list.append(trg)
        assert len(self.src_tensor_list) == len(self.trg_tensor_list)
        
        self.num_data = len(self.src_tensor_list)

    def __getitem__(self, index):
        src_encoded_dict = \
            self.src_tokenizer(
                self.src_tensor_list[index],
                max_length=self.src_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        src_input_ids = src_encoded_dict['input_ids'].squeeze(0)
        src_attention_mask = src_encoded_dict['attention_mask'].squeeze(0)

        trg = self.trg_tensor_list[index]

        return {'src_sequence' : src_input_ids, 'src_attention_mask' : src_attention_mask, 'trg_label' : trg}

    def __len__(self):
        return self.num_data