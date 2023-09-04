import argparse


MODEL = "bert-base-uncased"
DATASET = 'IMDB'
TASK = 'bert-base-uncased-classification'

file_dir = '/HDD/dataset/IMDB'
file_type ='csv'
model_path = 'models/save/bert-base-uncased-classification_IMDB_ver1.pt'

epochs = 100
lr = 0.1


def init_parser():
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    

    parser.add_argument('--model', default=MODEL, type=str)
    parser.add_argument('--file_dir', default=file_dir, type=str)
    parser.add_argument('--file_type', default=file_type, type=str)
    parser.add_argument('--model_path', default=model_path, type=str)

    parser.add_argument('--epochs', default=epochs, type=int,
                            help='epochs Default is 100')
    parser.add_argument('--lr', default=lr, type=int,
                            help='epochs Default is 0.1')
    
    parser.add_argument('--task', default=TASK, type=str)
    
        

        
    
    args = parser.parse_args()

    return args