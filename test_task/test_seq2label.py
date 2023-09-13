from models.model_init import model_init
import torch
from torch.utils.data import DataLoader
from utils.data_load import data_load
from models.dataset.dataset_init import dataset_init
import numpy as np
import wandb

from tqdm import tqdm
from utils import metric
from models.tokenizer.tokenizer_init import tokenizer_load

def test_seq2label(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Load
    test_src_list, test_trg_list = data_load(dataset_path=args.dataset_path, data_split_ratio=args.data_split_ratio,
                                            seed=args.seed, mode='test')
    args.num_classes = len(set(test_trg_list))

    model = model_init(args)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    # tokenizer init
    src_tokenizer = tokenizer_load(args)

    # Test dataset setting
    custom_dataset_dict = dict()
    custom_dataset_dict['src_tokenizer'] = src_tokenizer
    custom_dataset_dict['src_list'] = test_src_list['src_a']
    custom_dataset_dict['trg_list'] = test_trg_list['trg']
    test_dataset = dataset_init(args=args, dataset_dict=custom_dataset_dict)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=args.num_workers)
    # eval_accuracy = 0
    
    
    total_predictions = []
    total_labels = []
    
    for batch in tqdm(test_dataloader):
        # Input setting
        src_sequence = batch['src_sequence'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        trg_label = batch['trg_label'].to(device)
        
        with torch.no_grad(): # gradient 계산 context를 비활성화 ==> 필요한 메모리가 줄어들고 연산속도가 증가
            outputs = model(input_ids=src_sequence, attention_mask=src_attention_mask)

            logits = outputs.detach().cpu().numpy() 
            predictions = np.argmax(logits, axis=1)
            labels = trg_label.detach().cpu().numpy() 

            total_predictions.extend(predictions)
            total_labels.extend(labels)
            
    accuracy    = metric.get_accuracy(total_labels, total_predictions)
    recall      = metric.get_recall(total_labels, total_predictions)
    precision   = metric.get_precision(total_labels, total_predictions)
    f1          = metric.get_f1(total_labels, total_predictions)
    report      = metric.get_classification_report(total_labels, total_predictions)

    wandb.log({
            "accuracy" : accuracy,
            "recall" : recall,
            "precision" :precision,
            "f1":f1,                    
        })
    
    print(accuracy)
    print(recall)
    print(precision)
    print(f1)

    return accuracy, recall, precision, f1, report