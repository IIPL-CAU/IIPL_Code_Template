from transformers import AutoTokenizer

def tokenizer_load(args):
    if args.tokenizer == 'spm':
        pass
        '''
        spm.SentencePieceTrainer.Train(f'--input=imdb_review.txt --model_prefix=sentencepiece --vocab_size={vocab_size} --model_type=bpe --max_sentence_length={max_seq_length}')
        sp = spm.SentencePieceProcessor()
        vocab_file = "sentencepiece.model"
        '''
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=args.do_lower_case)
        
        if args.pretrained == False:
            def get_training_corpus(data):    # data : total_src_list['train'] = ["Hello, my dog is cute", "Hello, my dog is cute", 'I am a boy']
                return (data for i in range(0, len(data)))
                
            training_corpus = get_training_corpus(args.data)
            new_tokenizer = tokenizer.train_new_from_iterator(training_corpus, args.vocab_size)
            tokenizer = new_tokenizer
                
    return tokenizer