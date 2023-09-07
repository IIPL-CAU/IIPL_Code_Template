#from parser_script.iris_classification import init_parser
from parser_script.arg_parser import init_parser
from training import training
#from preprocessing import Preprocessing
from testing import testing

# Commit message template test용 @

if __name__=='__main__':
    args = init_parser()
    
    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)

    if args.testing:
        testing(args)