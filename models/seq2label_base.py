import torch
from torch import nn
from transformers import BertModel

class seq2label_base(nn.Module):
    def __init__(self, args):
        super(seq2label_base, self).__init__()
        """
        Initialize Seq2Label model

        Args:
            encoder_model_type (string): Encoder model's type
            decoder_model_type (string): Decoder model's type
            src_vocab_num (int): Source vocabulary number
            trg_vocab_num (int): Target vocabulary number
            isPreTrain (bool): Pre-trained model usage
            dropout (float): Dropout ratio
        """
        self.num_classes = args.num_classes
        self.bert_model_name = args.model
        
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_classes)

        # Linear Model Setting
        self.classify_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.classify_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.classify_linear2 = nn.Linear(self.d_embedding, self.num_classes)
    
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