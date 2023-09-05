# IIPL_Code_Template
This project is an Code Template project conducted by Chung-Ang University Intelligent Information Processing Lab. Our goal is to create a code template that can be helpful not only to new students entering our lab but also to everyone who wants to research deep learning.

### Dependencies

This code is written in Python. Dependencies include

* Python == 3.6
* PyTorch == 1.10.0
* Transformers (Huggingface) == 4.8.1
* NLG-Eval == 2.3.0 (https://github.com/Maluuba/nlg-eval)

### Usable Data
#### Neural Machine Translation
* WMT 2014 translation task **DE -> EN** (Coming soon...)
* WMT 2016 multimodal **DE -> EN** (Coming soon...)
* Korpora **EN -> KR** (Coming soon...)
#### Text Style Transfer
* Grammarly's Yahho Answer Formality Corpus **Informal -> Formal** (Coming soon...)
* Wiki Neutrality Corpus **Biased -> Neutral** (Coming soon...)
#### Summarization
* CNN & Daily Mail **News Summarization** (Coming soon...)
#### Classification
* IMDB **Sentiment Analysis** (--task=classification --data_name=IMDB)
* NSMC **Sentiment Analysis** (Coming soon...)
* Korean Hate Speech **Toxic Classification** (Coming soon...)

## Training

To train the model, add the training (--training) option. Currently, only the Transformer model is available, but RNN and Pre-trained Language Model will be added in the future.

```
python main.py --training
```