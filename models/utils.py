from transformers import AutoConfig, AutoModel

def encoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    encoder = basemodel.encoder

    return encoder, model_config

def decoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    decoder = basemodel.decoder

    return decoder, model_config