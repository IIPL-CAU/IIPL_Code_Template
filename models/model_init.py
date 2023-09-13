from models.seq2label_base import seq2label_base
from models.seq2seq_base import seq2seq_base
from transformers import AutoModelForSequenceClassification

def model_init(args):
    if args.task in ['single_text_classification', 'multi_text_classification', 'sentiment_analysis']:
        model = seq2label_base()

    if args.task in ['machine_translation', 'text_style_transfer', 'summarization']:
        model = seq2seq_base()

    if args.task in ['image_classification']:
        raise NotImplementedError("")
        # model = image_classification_base()

    return model