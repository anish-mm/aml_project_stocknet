"""
This file contains code to create the Market Information Encoder (MIE)
part of the stocknet architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import sys

from .model_utils import sequence_mask, get_last_seq_items
from .model_utils import PackedDropout

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
        word_table_init : numpy array - (vocab_size, word_embed_size)
            word lookup table
        """
        # for now, assume word_table_init is a numpy array
        super(WordEmbedder, self).__init__()
        self.word_table = torch.tensor(word_table_init)
        self.word_table.requires_grad = False
        # self.word_table = torch.tensor(word_table_init, requires_grad=False)

    def forward(self, words):
        """Forward pass to substitute word indices with corresponding embeddings
        from the lookup table.

        Parameters
        ----------
        word : IntTensor or LongTensor - (batch_size, max_valid_n_days, max_valid_n_msgs, max_valid_n_words)
            word indices

        Returns
        -------
        word_embed : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs, max_valid_n_words, word_embed_size)
            input to Message Embedding Layer
        """        
        # PyTorch doesn't have an equivalent for `tf.nn.embedding_lookup`.
        # So we use torch.`index_select` which takes only vector indices.
        idx_shape = words.shape
        indexed = torch.index_select(self.word_table, dim=0,
                    index=words.reshape(-1).to(dtype=torch.int64))
        return indexed.reshape(*idx_shape, -1)

class MessageEmbedder(nn.Module):
    """
    Combine the word embeddings in the message to get one embedding
    per message. Performs functions of `_create_msg_embed_layer_in` and 
    `_create_msg_embed_layer` from the original code.
    """
    def __init__(self, **kwargs):
        super(MessageEmbedder, self).__init__()
        # self.use_in_bn = kwargs["use_in_bn"]
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
        self.rnn = nn.Sequential(
            rnn(
                input_size=self.word_embed_size,
                hidden_size=self.mel_h_size,
                batch_first=True,
                bidirectional=True,
            ),
            PackedDropout(self.dropout_mel),
        )
        
        self.pack_padded_sequence = nn.utils.rnn.pack_padded_sequence
        self.pad_packed_sequence = nn.utils.rnn.pad_packed_sequence

    
    def forward(self, word_embed, n_words, ss_indices, n_msgs):
        """Combine word-level embeddings to get one embedding per tweet.

        Parameters
        ----------
        word_embed : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs, max_valid_n_words, word_embed_size)
            input to Message Embedding Layer
        n_words : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs)
            stores number of words in each message
        ss_indices : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs)
            stores indices of stock symbol (ss) in the tweets.
        n_msgs : tensor - (batch_size, max_valid_n_days)
            array recording number of tweets each day

        Returns
        -------
        msg_embed : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs, msg_embed_size)
            tweet data with message-level embeddings
        """        
        # batch normalization skipped for now. Original paper doesn't use it
        # in the main submission as per given config file. Also, pytorch doesn't
        # offer a built-in function to do this directly at the moment.
        mel_in = self.mel_in_dropout(word_embed)
        device = mel_in.device.type
        msg_embed_shape = (*mel_in.shape[:-2], self.mel_h_size)
        msg_embed = torch.zeros(*msg_embed_shape, device=device)
        # countsample = 0
        for i, (sample, sample_lengths) in enumerate(zip(mel_in, n_words)):
            # countsample += 1
            # countday = 0
            for j, (day, day_lengths) in enumerate(zip(sample, sample_lengths)):
                # countday += 1
                # try:
                #     n = ((day_lengths == 0).nonzero(as_tuple=True))[0][0]
                # except:
                #     n = day_lengths.shape[0]
                n = n_msgs[i, j]
                if n == 0:
                    # no messages for this day. This also means we have
                    # exhausted valid days for this sample.
                    break

                input=day[:n]
                lengths=day_lengths[:n]
                ss_index = (ss_indices[i, j, :n] - 1).unsqueeze(-1).to(dtype=torch.int64)
                # print(f"n = {n}")
                    # input=day
                    # lengths=day_lengths
                    # ss_index = (ss_indices[i, j] - 1).unsqueeze(-1).to(dtype=torch.int64)
                # print(f"input = {day[:n]}")
                # print(f"n_days = {n_days}")
                # print(f"day_lengths={day_lengths}")
                packed_day = self.pack_padded_sequence(
                    input=input, lengths=lengths.to(device="cpu", dtype=torch.int64),
                    batch_first=True, enforce_sorted=False,
                )
                if device == 'cuda':
                    packed_day = packed_day.cuda()
                # print(f"packed_day is on {packed_day.data.device}")
                # print(f"me rnn in = {packed_day}")
                rnn_out, _ = self.rnn(packed_day.float())
                # print(f"rnn_out is on {rnn_out.data.device}")
                # print(f"me rnn out = {rnn_out}")
                # we have a size issue here.
                padded_rnn_out, _ = self.pad_packed_sequence(rnn_out, batch_first=True)
                # print(f"padded_rnn_out is on {padded_rnn_out.device}")
                tweet_index = torch.arange(ss_index.shape[0]).unsqueeze(-1)
                # print(f"tweet_index is on {tweet_index.device}")
                # print(f"types = {ss_index.dtype, tweet_index.dtype}")
                # msg_embed_day_combined = get_last_seq_items(rnn_out, day_lengths)
                # padded_rnn
                # print(f"ss_index is on {ss_index.device}")
                msg_embed_day_combined = padded_rnn_out[tweet_index, ss_index].squeeze(1)
                # print(f"msg_embed_day_combined={msg_embed_day_combined}")
                # print(f"msg_embed_day_combined is on {msg_embed_day_combined.device}")
                mel_h_f = msg_embed_day_combined[:, :self.mel_h_size]
                mel_h_b = msg_embed_day_combined[:, self.mel_h_size:]
                msg_embed_day = (mel_h_f + mel_h_b) / 2
                # print(f"me msg_embed_day = {msg_embed_day}")
                msg_embed[i, j, :n] = msg_embed_day
                # if msg_embed_day.isnan().any():
                #     print(f"nan found in sample {i} - day {j}")
                #     sys.exit(1)
                # else:
                #     print(f"no nan in sample {i} - day {j}")
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
        self.dropout_ce = kwargs["dropout_ce"]

        self.proj_u = nn.Sequential(
            nn.Linear(self.msg_embed_size, self.msg_embed_size, bias=False),
            nn.Tanh(),
        )

        self.u = nn.Linear(self.msg_embed_size, 1)
        self.u_soft = nn.Softmax(dim=-1)
        self.corpus_embed_dropout = nn.Dropout(self.dropout_ce)
    
    def forward(self, msg_embed, n_msgs):
        """Forward pass that combines messages to create one embedding
        per day through an attention mechanism.

        Parameters
        ----------
        msg_embed : tensor - (batch_size, max_valid_n_days, max_valid_n_msgs, msg_embed_size)
            tweet data with message level embeddings
        n_msgs : tensor - (batch_size, max_valid_n_days)
            array recording number of tweets each day
        
        Returns
        -------
        corpus_embed : tweet data with day-level embeddings
            tensor - (batch_size, max_valid_n_days, msg_embed_size)
        """        
        # print(f"msg_embed is on {msg_embed.device}")
        # print("In corpus embedder")
        # if msg_embed.isnan().any():
        #     print(f"nan in msg_embed input")
        # else:
        #     print(f"no nan in msg_embed input")
        proj_u = self.proj_u(msg_embed)
        # if proj_u.isnan().any():
        #     print(f"nan in proj_u")
        # else:
        #     print(f"no nan in proj_u")
        u = self.u(proj_u).squeeze(dim=-1) # (batch_size, max_n_days, max_n_msgs)
        # if u.isnan().any():
        #     print(f"nan in u")
        # else:
        #     print(f"no nan in u")
        mask_msgs = sequence_mask(n_msgs)
        masked_score = torch.where(mask_msgs, u, torch.tensor(-np.inf, device=u.device))
        # if masked_score.isnan().any():
        #     print(f"nan in masked_score")
        # else:
        #     print(f"no nan in masked_score")
        # -inf will result in 0 when softmax is applied.
        # but if last dim. has all -inf, then the dr of softmax bcomes 0, results in nan.
        # to account for this, we replace all nans with 0s after softmax.
        u = self.u_soft(masked_score)
        u = torch.where(u.isnan(), torch.zeros_like(u), u)
        # if u.isnan().any():
        #     print(f"nan in u2")
        # else:
        #     print(f"no nan in u2")
        u = u.unsqueeze(dim=-2)

        corpus_embed = torch.matmul(u, msg_embed)
        corpus_embed = corpus_embed.squeeze(dim=-2)
        corpus_embed = self.corpus_embed_dropout(corpus_embed)
        # if corpus_embed.isnan().any():
        #     print("nan in corpus embedder output")
        # else: 
        #     print("no nan in corpus embedder output")
        return corpus_embed

class BatchCompacter(nn.Module):
    def __init__(self):
        super(BatchCompacter, self).__init__()

    def forward(self, prices, words, n_days, n_msgs, n_words, ss_indices):
        """[summary]

        Parameters
        ----------
        prices : tensor - (batch_size, max_n_days, price_size)
            normalized price data
        words : tensor - (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)
            matrix of tweet data with indices of used words in vocabulary.
        n_days : tensor - (batchsize,)
            number of valid trading days in each sample
        n_msgs : tensor - (batch_size, max_n_days)
            number of tweets each day
        n_words : tensor - (batch_size, max_n_days, max_n_msgs)
            stores number of words in each tweet
        ss_indices : tensor - (batch_size, max_n_days, max_n_msgs)
            indices of the stock symbol (ss) in the tweets.

        Returns
        -------
        dict
            compact versions of inputs and updated maximums.
        """        
        max_valid_days = n_days.max()
        valid_n_msgs = n_msgs[:, :max_valid_days]
        
        max_valid_msgs = valid_n_msgs.max()
        valid_n_words = n_words[:, :max_valid_days, :max_valid_msgs]

        max_valid_words = valid_n_words.max()
        
        valid_words = words[:, :max_valid_days, :max_valid_msgs, :max_valid_words]
        valid_prices = prices[:, :max_valid_days]
        valid_ss_indices = ss_indices[:, :max_valid_days, :max_valid_msgs]

        return {
            "prices": valid_prices,
            "words": valid_words,
            "n_days": n_days,
            "max_n_days": max_valid_days,
            "n_msgs": valid_n_msgs,
            "max_n_msgs": max_valid_msgs,
            "n_words": valid_n_words,
            "max_n_words": max_valid_words,
            "ss_indices": valid_ss_indices,
        }

class MIE(nn.Module):
    def __init__(self, **kwargs):
        super(MIE, self).__init__()
        
        self.word_table_init = kwargs["word_table_init"]

        self.price_size = kwargs["price_size"]
        # self.word_embed_size = kwargs["word_embed_size"]
        # self.msg_embed_size = kwargs["mel_h_size"]
        self.corpus_embed_size = kwargs["mel_h_size"]
        # self.mel_h_size = kwargs["mel_h_size"]

        # self.max_n_msgs = kwargs["max_n_msgs"]
        self.variant_type = kwargs["variant_type"]
        # self.dropout_ce = kwargs["dropout_ce"] # (1 - keep_prob) for dropout to corpus embedding.
        # self.use_in_bn = kwargs["use_in_bn"] # batch normalization in MEL input. unused for now.

        self.x = None
        if self.variant_type == "tech":
            x_size = self.price_size
        else:
            self.batch_compacter = BatchCompacter()
            self.word_embedder = WordEmbedder(self.word_table_init)
            self.message_embedder = MessageEmbedder(**kwargs)
            self.corpus_embedder = CorpusEmbedder(**kwargs)

            if self.variant_type == "fund":
                x_size = self.corpus_embed_size
            else:
                x_size = self.corpus_embed_size + self.price_size
        self.x_size = x_size


    def forward(self, prices, words, ss_indices, n_days, n_msgs, n_words):
        """Forward pass for MIE.

        Parameters
        ----------
        price : tensor - (batch_size, max_n_days, price_size)
            normalized price data  
        word : tensor - (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)
            matrix of tweet data with indices of used words in vocabulary.
        n_days: tensor - (batchsize,)
            number of days per sample
        n_msgs : tensor - (batch_size, max_n_days)
            array recording number of tweets each day
        ss_indices : tensor (batch_size, max_n_days, max_n_msgs)
            index of stock symbol (ss) in the tweet.
        n_words : tensor - (batch_size, max_n_days, max_n_msgs)
            stores number of words in each tweet

        Returns
        -------
        x : tensor - (batch_size, max_valid_n_days, x_size)
            output of MIE; input to Variational Movement Decoder (VMD)
        """        

        # print(f"prices = {prices}")
        max_n_days = n_days.max() 
        if self.variant_type == "tech":
            # TECHNICALANALYST: Doesn't use tweets.
            x = prices[:, :max_n_days]
        else:
            # Remaining models use tweets.
            compacted = self.batch_compacter(
                prices=prices, words=words, 
                n_days=n_days, n_msgs=n_msgs,
                n_words=n_words, ss_indices=ss_indices,
            )
            # print(f"compacted - {compacted}")
            word_embed = self.word_embedder(compacted["words"])
            # print(f"word_embed = {word_embed}")
            msg_embed = self.message_embedder(
                word_embed=word_embed, n_words=compacted["n_words"],
                ss_indices=compacted["ss_indices"],
                n_msgs=compacted["n_msgs"],
            )
            # print(f"msg_embed = {msg_embed}")
            corpus_embed = self.corpus_embedder(msg_embed, compacted["n_msgs"])
            # print(f"corpus_embed = {corpus_embed}")
            if self.variant_type == "fund":
                # FUNDAMENTALANALYST: Only uses tweets.
                x = corpus_embed
            else:
                # HEDGEFUNDANAYST / INDEPENDENTANALYST: use both.
                x = torch.cat((corpus_embed, compacted["prices"]), dim=2)
        return x, max_n_days







        