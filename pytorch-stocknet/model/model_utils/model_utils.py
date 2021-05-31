"""
Utility functions for creating the stocknet model.
"""
import torch
from torch.nn.modules import rnn
from torch.nn.utils.rnn import PackedSequence

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """
    :param lengths: array with lengths as the last dimension.
    For instance, array of shape (batch_size, num_days) where
    each value represents the number of tweets present on that
    day.
    refer `this discussion <https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/4>`

    :type lengths: tensor

    :return: mask
    :rtype: tensor of shape (*lengths.shape, maxlen)
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def get_last_seq_items(packed, lengths):
    """
    refer `this discussion <https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7>`
    """
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    ))
    sorted_lengths = lengths[packed.sorted_indices]
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


def kl_normal_normal(ploc, pscale, qloc, qscale):
    """kl-divergence between two normal distributions.
    taken from `here <https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence>`

    Parameters
    ----------
    ploc : tensor - (N1, N2, ..., Nk)
        mean of p distribution
    pscale : tensor - (N1, N2, ..., Nk)
        std of p distribution
    qloc : tensor - (N1, N2, ..., Nk)
        mean of q distribution 
    qscale : tensor - (N1, N2, ..., Nk)
        std of q distribution

    Returns
    -------
    tensor - (N1, N2, ..., Nk)
        kl-divergences.
    """    
    var_ratio = (pscale / qscale).pow(2)
    t1 = ((ploc - qloc) / qscale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

class PackedDropout(torch.nn.Module):   
    """
    Dropout intended to be used with RNN/GRU/LSTM
    when these models are run on PackedSequence objects.
    """ 
    def __init__(self, p):
        """ Initialize the internal dropout layer.

        Parameters
        ----------
        p : float
            drop probability
        """        
        super(PackedDropout, self).__init__()

        self.p = p
        self.dropout = torch.nn.Dropout(p)
    
    def forward(self, rnn_out):
        """Forward pass for the dropout.

        Parameters
        ----------
        rnn_out : Tuple (output, hn, ..):
            output : The packed sequence to apply dropout to.
            hn, ... : Other outputs like hidden state values returned.

        Returns
        -------
        PackedSequence
            result after applying dropout.
        """        
        packed_seq, *others = rnn_out
        assert isinstance(packed_seq, PackedSequence), \
            "rnn output should be a PackedSequence instance."
        
        data, *rest = packed_seq
        data = self.dropout(data)
        return (PackedSequence(data, *rest), *others)
    