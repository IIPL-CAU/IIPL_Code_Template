from models.ANN import ANN


def model_init(args):
    if args.model == "ANN":
        return ANN(args)