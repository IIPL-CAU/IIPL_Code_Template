from parser_script.iris_classification import init_parser
from training import training
from preprocessing import Preprocessing
from testing import testing



if __name__=='__main__':
    args = init_parser()
    
    
    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)

    if args.testing:
        testing(args)

    
