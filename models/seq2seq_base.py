
import math
from collections import defaultdict
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
# Import Huggingface
from transformers import AutoTokenizer
# Import Custom Modules
from .utils import encoder_model_setting, decoder_model_setting

class seq2seq_base(nn.Module):
    def __init__(self, encoder_model_type: str = 't5-base', decoder_model_type: str = 't5-base',
                 src_vocab_num: int = 32000, trg_vocab_num: int = 32000,
                 isPreTrain: bool = True, dropout: float = 0.3):
        super(seq2seq_base, self).__init__()

        """
        Initialize Seq2Seq model

        Args:
            encoder_model_type (string): Encoder model's type
            decoder_model_type (string): Decoder model's type
            src_vocab_num (int): Source vocabulary number
            trg_vocab_num (int): Target vocabulary number
            isPreTrain (bool): Pre-trained model usage
            dropout (float): Dropout ratio
        """
        self.isPreTrain = isPreTrain
        self.dropout = nn.Dropout(dropout)

        # Encoder model setting
        self.encoder_model_type = encoder_model_type
        encoder, encoder_model_config = encoder_model_setting(encoder_model_type, self.isPreTrain)

        self.encoder = encoder

        # Decoder model setting
        self.decoder_model_type = decoder_model_type
        decoder, decoder_model_config = decoder_model_setting(decoder_model_type, self.isPreTrain)

        self.vocab_num = trg_vocab_num
        self.d_hidden = decoder_model_config.d_model
        self.d_embedding = int(self.d_hidden / 2)

        self.decoder = decoder

        # Linear Model Setting
        self.decoder_linear = nn.Linear(self.d_hidden, self.d_embedding)
        self.decoder_norm = nn.LayerNorm(self.d_embedding, eps=1e-12)
        self.decoder_linear2 = nn.Linear(self.d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_type)

        self.pad_idx = self.tokenizer.pad_token_id
        self.bos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.decoder_start_token_id = decoder_model_config.decoder_start_token_id

    def encode(self, src_input_ids, src_attention_mask=None):
        if src_input_ids.dtype == torch.int64:
            encoder_out = self.encoder(input_ids=src_input_ids,
                                       attention_mask=src_attention_mask)
        else:
            encoder_out = self.encoder(inputs_embeds=src_input_ids,
                                       attention_mask=src_attention_mask)
        encoder_out = encoder_out['last_hidden_state'] # (batch_size, seq_len, d_hidden)

        return encoder_out

    def decode(self, trg_input_ids, encoder_hidden_states=None, encoder_attention_mask=None):
        decoder_input_ids = shift_tokens_right(
            trg_input_ids, self.pad_idx, self.decoder_start_token_id
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )

        decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, seq_len, d_hidden)
        print(decoder_outputs.size())
        decoder_outputs = self.dropout(F.gelu(self.decoder_linear(decoder_outputs)))
        decoder_outputs = self.decoder_linear2(self.decoder_norm(decoder_outputs))

        return decoder_outputs

    def generate(self, decoding_dict:dict = dict(), encoder_hidden_states=None, encoder_attention_mask=None,
                 beam_size=3, beam_alpha=0.7, repetition_penalty=0.7):

        # Input, output setting
        device = encoder_hidden_states.device
        batch_size = encoder_hidden_states.size(0)
        src_seq_size = encoder_hidden_states.size(1)
        every_batch = torch.arange(0, beam_size * batch_size, beam_size, device=device)

        # Encoder hidden state expanding
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1) # (batch_size, 1, seq_len, d_hidden)
        encoder_hidden_states = encoder_hidden_states.repeat(1, beam_size, 1, 1) # (batch_size, beam_size, seq_len, d_hidden)
        encoder_hidden_states = encoder_hidden_states.view(-1, src_seq_size, self.d_hidden) # (batch_size * beam_size, seq_len, d_hidden)

        # Scores save vector & decoding list setting
        scores_save = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
        top_k_scores = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
        complete_seqs = defaultdict(list)
        complete_ind = set()

        # Decoding start token setting
        seqs = torch.tensor([[self.decoder_start_token_id]], dtype=torch.long, device=device)
        seqs = seqs.repeat(beam_size * batch_size, 1).contiguous() # (batch_size * beam_size, 1)

        for step in range(self.src_max_len):
            # Decoding sentence
            decoder_outputs = self.decoder(
                input_ids=seqs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            decoder_outputs = decoder_outputs['last_hidden_state']

            # Score calculate
            scores = F.gelu(self.decoder_linear(decoder_outputs[:,-1])) # (batch_size * k, d_embedding)
            scores = self.decoder_linear2(self.decoder_norm(scores)) # (batch_size * k, vocab_num)
            scores = F.log_softmax(scores, dim=1) # (batch_size * k, vocab_num)

            # Add score
            scores = top_k_scores.expand_as(scores) + scores  # (batch_size * k, vocab_num)
            if step == 0:
                scores = scores[::beam_size] # (batch_size, vocab_num)
                scores[:, self.eos_idx] = float('-inf') # set eos token probability zero in first step
                top_k_scores, top_k_words = scores.topk(beam_size, 1, True, True)  # (batch_size, k) , (batch_size, k)
            else:
                top_k_scores, top_k_words = scores.view(batch_size, -1).topk(beam_size, 1, True, True)

            # Previous and Next word extract
            prev_word_inds = top_k_words // self.vocab_num # (batch_size * k, out_seq)
            next_word_inds = top_k_words % self.vocab_num # (batch_size * k, out_seq)
            top_k_scores = top_k_scores.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            top_k_words = top_k_words.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, beam_size).view(-1)] # (batch_size * k, out_seq)
            seqs = torch.cat([seqs, next_word_inds.view(beam_size * batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

            # Find and Save Complete Sequences Score
            if self.eos_idx in next_word_inds:
                eos_ind = torch.where(next_word_inds.view(-1) == self.eos_idx)
                eos_ind = eos_ind[0].tolist()
                complete_ind_add = set(eos_ind) - complete_ind
                complete_ind_add = list(complete_ind_add)
                complete_ind.update(eos_ind)
                if len(complete_ind_add) > 0:
                    scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                    for ix in complete_ind_add:
                        complete_seqs[ix] = seqs[ix].tolist()

        # If eos token doesn't exist in sequence
        if 0 in scores_save:
            score_save_pos = torch.where(scores_save == 0)
            for ix in score_save_pos[0].tolist():
                complete_seqs[ix] = seqs[ix].tolist()
            scores_save[score_save_pos] = top_k_scores[score_save_pos]

        # Beam Length Normalization
        lp = torch.tensor([len(complete_seqs[i]) for i in range(batch_size * beam_size)], device=device)
        lp = (((lp + beam_size) ** beam_alpha) / ((beam_size + 1) ** beam_alpha)).unsqueeze(1)
        scores_save = scores_save / lp

        # Predicted and Label processing
        _, ind = scores_save.view(batch_size, beam_size, -1).max(1)
        ind_expand = ind.view(-1) + every_batch
        predicted = [complete_seqs[i] for i in ind_expand.tolist()]

        return torch.tensor(predicted, device=device)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids