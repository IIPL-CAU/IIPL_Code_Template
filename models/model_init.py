from models.ANN import ANN
from transformers import AutoModelForSequenceClassification

def model_init(args):
    if args.model == "ANN":
        return ANN(args)
    if args.model == "bert-base-uncased":
        print("o")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        return model