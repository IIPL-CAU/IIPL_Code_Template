import argparse

MODEL = 'bert-base-uncased'
tokenizer = 'bert-base-uncased'
TASK = 'single_text_classification'
model_path = 'models/save/bert_base_uncased_classification_IMDB_ver1.pt'

file_dir = 'imdb'
file_type ='csv'
DATASET = 'IMDB'

epochs = 4
lr = 2e-5
optim_type = 'AdamW'

scheduler_type = 'get_linear_schedule_with_warmup'
batch_size = 16



def init_parser():
    parser = argparse.ArgumentParser(description='Parsing Method')

    parser.add_argument('--wandb', default='IMDB', type=str, help='wandb run name')
    # Task setting
    parser.add_argument('--task', default=TASK, type=str)
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--val_ratio', default=0.2, type=float,
                            help='val_ratio Default is 0.2')
    parser.add_argument('--batch_size', default=batch_size, type=int)

    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--dataset_path', default=file_dir, type=str)
    parser.add_argument('--file_type', default=file_type, type=str)
    parser.add_argument('--model_path', default=model_path, type=str)

    parser.add_argument('--optim_type', default=optim_type, type=str)
    parser.add_argument('--weight_decay', default=0.01, type=float,
                            help='epochs Default is 0.01')
    parser.add_argument('--scheduler_type', default=scheduler_type, type=str)

    parser.add_argument('--epochs', default=epochs, type=int,
                            help='epochs Default is 100')
    parser.add_argument('--lr', default=lr, type=float,
                            help='epochs Default is 0.1')

    #-------- Additional argument! Need to be refactored ---------#
    parser.add_argument('--data_split_ratio', default=[0.8,0.1,0.1], type=list)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--tokenizer', default=tokenizer, type=str)
    parser.add_argument('--do_lower_case', default=True, type=bool)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--min_len', default=10, type=int)
    parser.add_argument('--src_max_len', default=300, type=int)
    parser.add_argument('--trg_max_len', default=300, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()

    return args