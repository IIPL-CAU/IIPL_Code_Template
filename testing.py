
from test_task.test_seq2label import test_seq2label
from test_task.test_seq2seq import test_seq2seq


def testing(args):
    if args.task == 'single_text_classification':
        test_seq2label(args)
    if args.task == 'machine_translation':
        test_seq2seq(args)

            
            
        
