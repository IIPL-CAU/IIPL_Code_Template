import argparse
from models.dataset.seq2label_dataset import Seq2Label_CustomDataset
from models.dataset.seq2seq_dataset import Seq2Seq_CustomDataset

def dataset_init(args:argparse.Namespace = None, dataset_dict:dict = None):
    if args.task == "single_text_classification":
        return Seq2Label_CustomDataset(src_tokenizer=dataset_dict['src_tokenizer'],
                                       src_list=dataset_dict['src_list'], trg_list=dataset_dict['trg_list'],
                                       min_len=args.min_len, src_max_len=args.src_max_len)

    if args.task == "machine_translation":
        return Seq2Seq_CustomDataset(src_tokenizer=dataset_dict['src_tokenizer'], trg_tokenizer=dataset_dict['trg_tokenizer'],
                                     src_list=dataset_dict['src_list'], trg_list=dataset_dict['trg_list'],
                                     min_len=args.min_len, src_max_len=args.src_max_len, trg_max_len=args.trg_max_len)