from models.model_init import model_init
import torch
from utils.data_load import data_load
import numpy as np

from tqdm import tqdm
import metric



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def testing(args):
    
    if args.task == 'bert-base-uncased-classification model':
        model = model_init(args)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()  # Dropout, Batchnorm등의 기능을 비활성화
        # eval_accuracy = 0

        for encoded, label in tqdm(test_dataloader):

            # input_ids = batch['input_ids'].to(device)
            # input_mask = batch['attention_mask'].to(device)
            # labels = batch['label'].to(device)
            
            with torch.no_grad(): # gradient 계산 context를 비활성화 ==> 필요한 메모리가 줄어들고 연산속도가 증가
                outputs = model(**encoded, labels=label).to(device)
                # outputs = model(input_ids, 
                #         token_type_ids=None, 
                #         attention_mask=input_mask)
                
                logits = outputs.logits

            logits = logits.detach().cpu().numpy() 
            predictions = np.argmax(logits, axis=1)
            labels = labels.detach().cpu().numpy() 
            
            accuracy = metric.get_accuracy(labels, predictions)
            recall_score = metric.get_recall(labels, predictions)
            precision_score = metric.get_precision(labels, predictions)
            f1_score = metric.get_f1(labels, predictions)
            report = metric.get_classification_report(labels, predictions)
            

        return accuracy, recall_score, precision_score, f1_score, report
        
