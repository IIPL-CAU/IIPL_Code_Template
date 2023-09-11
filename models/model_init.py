from models.ANN import ANN
from models.BERTClassifier import BERTClassifier
from transformers import AutoModelForSequenceClassification

def model_init(args):
    if args.model == "ANN":
        return ANN(args)
    if args.model == "bert-base-uncased":
        model = BERTClassifier(args)
        return model