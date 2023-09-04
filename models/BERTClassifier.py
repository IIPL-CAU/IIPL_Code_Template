import torch
from torch import nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, args):
        super(BERTClassifier, self).__init__()
        self.num_classes = args.num_classes
        self.bert_model_name = args.bert_model_name
        
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_classes)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def classify(self, outputs):
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

    def forward(self, input_ids, attention_mask):
        outputs = self.encode(input_ids, attention_mask)
        logits = self.classify(outputs)
        return logits