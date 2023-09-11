import torch.nn as nn

def get_criterion(criterion_type:str=None) -> nn.Module:
    ''' 
        criterion_type (str): criterion type {cross_entropy, ...}
    '''
    criterion_type = criterion_type.lower()

    if criterion_type in ["l1", "l1_loss"]:
        return nn.L1Loss()

    elif criterion_type in ["mse", "mse_loss"]:
        return nn.MSELoss()

    elif criterion_type in ["cross_entropy", "crossentropy", "ce", "cross_entropy_loss", "crossentropyloss"]:
        return nn.CrossEntropyLoss()
    
    elif criterion_type in ["ctc", "ctc_loss"]:
        return nn.CTCLoss()

    elif criterion_type in ["nll", "nll_loss"]:
        return nn.NLLLoss()
    
    elif criterion_type in ["poisson_nll", "poisson_nll_loss", "poissonnll", "poissonnllloss"]:
        return nn.PoissonNLLLoss()
    
    elif criterion_type in ["caussian_nll", "caussian_nll_loss", "caussiannll", "caussiannllloss"]:
        return nn.GaussianNLLLoss()

    elif criterion_type in ["kl_div", "kl_div_loss", "kldiv", "kldivloss"]:
        return nn.KLDivLoss()

    elif criterion_type in ["bce", "bce_loss"]:
        return nn.BCELoss()
    
    elif criterion_type in ["bce_with_logits", "bce_with_logits_loss", "becwithlogits", "becwithlogitsloss"]:
        return nn.BCEWithLogitsLoss()
     
    elif criterion_type in ["margin_ranking", "margin_ranking_loss", "marginranking", "marginrankingloss"]:
        return nn.MarginRankingLoss()
    
    elif criterion_type in ["hinge_embedding", "hinge_embedding_loss", "hingeembedding", "hingeembeddingloss"]:
        return nn.HingeEmbeddingLoss()
    
    elif criterion_type in ["multi_label_margin", "multi_label_margin_loss", "multilabelmargin", "multilabelmarginloss"]:
        return nn.MultiLabelMarginLoss()
    
    elif criterion_type in ["huber", "huber_loss"]:
        return nn.HuberLoss()

    elif criterion_type in ["smooth_l1", "smooth_l1_loss", "smoothl1", "smoothl1loss"]:
        return nn.SmoothL1Loss()

    elif criterion_type in ["soft_margin", "soft_margin_loss", "softmargin", "softmarginloss"]:
        return nn.SoftMarginLoss()

    elif criterion_type in ["multi_label_soft_margin", "multi_label_soft_margin_loss", "multilabelsoftmargin", "multilabelsoftmarginloss"]:
        return nn.MultiLabelSoftMarginLoss()

    elif criterion_type in ["cosine_embedding", "cosine_embedding_loss", "cosineembedding", "cosineembeddingloss"]:
        return nn.CosineEmbeddingLoss()
    
    elif criterion_type in ["multi_margin", "multi_margin_loss", "multimargin", "multimarginloss"]:
        return nn.MultiMarginLoss()
    
    elif criterion_type in ["triplet_margin", "triplet_margin_loss", "tripletmargin", "tripletmarginloss"]:
        return nn.TripletMarginLoss()

    elif criterion_type in ["triplet_margin_with_distance_loss", "triplet_margin_with_distance", "tripletmarginwithdistance", "tripletmarginwithdistanceloss"]:
        return nn.TripletMarginWithDistanceLoss()

    else:
        raise ValueError("Unknown criterion type: {}".format(criterion_type))
