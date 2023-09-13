from train_task.train_seq2label import train_seq2label
from train_task.train_seq2seq import train_seq2seq


def training(args):
    if args.task =='single_text_classification':
        train_seq2label(args)
        
    if args.task =='machine_translation':
        train_seq2seq(args)
