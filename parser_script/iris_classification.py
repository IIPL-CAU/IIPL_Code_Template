import argparse


MODEL = "ANN"
DATASET = 'iris'
TASK = 'iris_classification'

file_dir = '/HDD/juhyoung/iris/Iris.csv'
file_type ='csv'
model_path = 'models/save/iris_classification_ver1.pt'

epochs = 100
lr = 0.1

in_features = 4
h1 = 8
h2 = 9
out_features = 3

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
    
    if TASK == 'iris_classification':
        parser.add_argument('--task', default=TASK, type=str)
        parser.add_argument('--in_features', default=in_features, type=int, 
                            help='in_features Default is 4')
        parser.add_argument('--h1', default=h1, type=int, 
                            help='h1 Default is 8')
        parser.add_argument('--h2', default=h2, type=int, 
                            help='h2 Default is 9')
        parser.add_argument('--out_features', default=out_features, type=int, 
                            help='out_features Default is 3')
        
        

        
    
    args = parser.parse_args()

    return args