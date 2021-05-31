"""
This file contains code to create the Market Information Encoder (MIE)
part of the stocknet architecture.
"""

import numpy as np
import torch
import torch.nn as nn

from .model_utils import sequence_mask, get_last_seq_items

class WordEmbedder(nn.Module):
    """
    Look up embeddings of words to get word-level embeddings for the
    tweet data. Performs similar functionality to `_build_embeds` form
    the main code.
    """
    def __init__(self, word_table_init):
        """[summary]

        Parameters
        ----------
        word_table_init : numpy array - (vocab_size, word_embedding_size)
            word lookup table
        """
        # for now, assume word_table_init is a numpy array
        super(WordEmbedder, self).__init__()
        self.word_table = torch.tensor(word_table_init, requires_grad=False)

    def forward(self, word):
        """Forward pass to substitute word indices with corresponding embeddings
        from the lookup table.

        Parameters
        ----------
        word : IntTensor or LongTensor - (batch_size, max_n_days, max_n_msgs, max_n_words)
            word indices

        Returns
        -------
        word_embed : tensor - (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)
            input to Message Embedding Layer
        """        
        # PyTorch doesn't have an equivalent for `tf.nn.embedding_lookup`.
        # So we use torch.`index_select` which takes only vector indices.
        idx_shape = word.shape
        indexed = torch.index_select(self.word_table, dim=0,
                    index=word.reshape(-1))
        return indexed.reshape(*idx_shape, -1)

class MessageEmbedder(nn.Module):
    """
    Combine the word embeddings in the message to get one embedding
    per message. Performs functions of `_create_msg_embed_layer_in` and 
    `_create_msg_embed_layer` from the original code.
    """
    def __init__(self, *args, **kwargs):
        super(MessageEmbedder, self).__init__(**kwargs)
        self.use_in_bn = kwargs["use_in_bn"]
        self.dropout_mel_in = kwargs["dropout_mel_in"] # dropout for MEL input
        self.dropout_mel = kwargs["dropout_mel"] # dropout for MEL RNN/GRU/lstm cell
        self.word_embed_size = kwargs["word_embed_size"]
        self.mel_h_size = kwargs["mel_h_size"]
        self.mel_cell_type = kwargs["mel_cell_type"]

        # layers
        self.mel_in_dropout = nn.Dropout(self.dropout_mel_in)
        self.msg_embed_dropout = nn.Dropout(self.dropout_mel)

        if self.mel_cell_type == "ln-lstm":
            rnn = nn.LSTM
        elif self.mel_cell_type == "gru":
            rnn = nn.GRU
        else:
            rnn = nn.RNN
        """
        TODO:
            1. They use an LSTM cell which has batch-normalization. PyTorch
            does not offer an in-built alternative for this. Will need to 
            understand how this normalization works in order to implement it.
            One possible alternative is given at ` exe1023/LSTM_LN 
            <https://github.com/exe1023/LSTM_LN>`.
            2. We have not performed Batch-normalization on mel_in.
        """
        self.rnn = rnn(
                input_size=self.word_embed_size,
                hidden_size=self.mel_h_size,
                batch_first=True,
                dropout=self.dropout_mel,
                bidirectional=True,
            )
        self.pack_padded_sequence = nn.utils.rnn.pack_padded_sequence

    
    def forward(self, word_embed, ss_index):
        """Combine word-level embeddings to get one embedding per tweet.

        Parameters
        ----------
        word_embed : tensor - (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)
            input to Message Embedding Layer
        ss_index : tensor - (batch_size, max_n_days, max_n_msgs)
            stores number of words in each message

        Returns
        -------
        msg_embed : tensor - (batch_size, max_n_days, max_n_msgs, msg_embed_size)
            tweet data with message-level embeddings
        """        
        # batch normalization skipped for now. Original paper doesn't use it
        # in the main submission as per given config file. Also, pytorch doesn't
        # offer a built-in function to do this directly at the moment.
        mel_in = self.mel_in_dropout(word_embed)

        msg_embed_shape = (*mel_in.shape[:-2], self.mel_h_size)
        msg_embed = torch.zeros(*msg_embed_shape)
        for i, (sample, sample_lengths) in enumerate(zip(mel_in, ss_index)):
            for j, (day, day_lengths) in enumerate(zip(sample, sample_lengths)):
                packed_day = self.pack_padded_sequence(
                    input=day, lengths=day_lengths,
                    batch_first=True, enforce_sorted=False,
                )
                rnn_out, _ = self.rnn(packed_day)
                
                msg_embed_day_combined = get_last_seq_items(rnn_out, day_lengths)
                mel_h_f = msg_embed_day_combined[:, :self.mel_h_size]
                mel_h_b = msg_embed_day_combined[:, self.mel_h_size:]
                msg_embed_day = (mel_h_f + mel_h_b) / 2
                msg_embed[i, j] = msg_embed_day
        msg_embed = msg_embed.reshape(msg_embed_shape)

        return self.msg_embed_dropout(msg_embed)

class CorpusEmbedder(nn.Module):
    """
    This function implements the attention mechanism that assigns weights
    to the messages to create one embedding per day, as a weighted average
    of the individual messages.
    """
    def __init__(self, **kwargs):
        super(CorpusEmbedder, self).__init__()
        self.msg_embed_size = kwargs["mel_h_size"]
        self.max_n_msgs = kwargs["max_n_msgs"]
        self.dropout_ce = kwargs["dropout_ce"]

        self.proj_u = nn.Sequential(
            nn.Linear(self.msg_embed_size, self.msg_embed_size, bias=False),
            nn.Tanh(),
        )

        self.u = nn.Linear(self.msg_embed_size, 1)
        self.u_soft = nn.Softmax(dim=-1)
        self.corpus_embed = nn.Dropout(self.dropout_ce)
    
    def forward(self, msg_embed, n_msgs):
        """Forward pass that combines messages to create one embedding
        per day through an attention mechanism.

        Parameters
        ----------
        msg_embed : tensor - (batch_size, max_n_days, max_n_msgs, msg_embed_size)
            tweet data with message level embeddings
        n_msgs : tensor - (batch_size, max_n_days)
            array recording number of tweets each day

        Returns
        -------
        corpus_embed : tweet data with day-level embeddings
            tensor - (batch_size, max_n_days, msg_embed_size)
        """        
        proj_u = self.proj_u(msg_embed)
        u = self.u(proj_u).squeeze(dim=-1)
        mask_msgs = sequence_mask(n_msgs, maxlen=self.max_n_msgs)
        masked_score = torch.where(mask_msgs, u, torch.tensor(-np.inf))
        # -inf will result in 0 when softmax is applied.
        u = self.u_soft(masked_score)
        u = u.unsqueeze(dim=-2)

        corpus_embed = torch.matmul(u, msg_embed)
        corpus_embed = corpus_embed.squeeze(dim=-2)
        corpus_embed = self.corpus_embed(corpus_embed)

        return corpus_embed

class MIE(nn.Module):
    def __init__(self, **kwargs):
        super(MIE, self).__init__()
        
        self.word_table_init = kwargs["word_table_init"]

        self.price_size = kwargs["price_size"]
        self.word_embed_size = kwargs["word_embed_size"]
        self.msg_embed_size = kwargs["mel_h_size"]
        self.corpus_embed_size = kwargs["mel_h_size"]
        self.mel_h_size = kwargs["mel_h_size"]

        self.max_n_msgs = kwargs["max_n_msgs"]
        self.variant_type = kwargs["variant_type"]
        self.dropout_ce = kwargs["dropout_ce"] # (1 - keep_prob) for dropout to corpus embedding.
        # self.use_in_bn = kwargs["use_in_bn"] # batch normalization in MEL input?

        self.x = None
        if self.variant_type == "tech":
            x_size = self.price_size
        else:
            self.word_embedder = WordEmbedder(self.word_table_init)
            self.message_embedder = MessageEmbedder(**kwargs)
            self.corpus_embedder = CorpusEmbedder(**kwargs)

            if self.variant_type == "fund":
                x_size = self.corpus_embed_size
            else:
                x_size = self.corpus_embed_size + self.price_size
        self.x_size = x_size


    def forward(self, price, word, ss_index, n_msgs):
        """Forward pass for MIE.

        Parameters
        ----------
        price : tensor - (batch_size, max_n_days, price_size)
            normalized price data  
        word : tensor - (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)
            matrix of tweet data with indices of used words in vocabulary.
        ss_index : tensor - (batch_size, max_n_days, max_n_msgs)
            stores number of words in each tweet
        n_msgs : tensor - (batch_size, max_n_days)
            array recording number of tweets each day

        Returns
        -------
        x : tensor - (batch_size, max_n_days, x_size)
            output of MIE; input to Variational Movement Decoder (VMD)
        """        
        if self.variant_type == "tech":
            # TECHNICALANALYST: Doesn't use tweets.
            x = price
        else:
            # Remaining models use tweets.
            word_embed = self.word_embedder(word)
            msg_embed = self.message_embedder(word_embed, ss_index)
            corpus_embed = self.corpus_embedder(msg_embed, n_msgs)
            
            if self.variant_type == "fund":
                # FUNDAMENTALANALYST: Only uses tweets.
                x = corpus_embed
            else:
                # HEDGEFUNDANAYST / INDEPENDENTANALYST: use both.
                x = torch.cat((corpus_embed, price), dim=2)
        return x







        