"""
This file contains code to create the Variational Movement Decoder (VMD)
part of the stocknet architecture.
"""

from torch.sparse import softmax
from model.model_utils.model_utils import kl_normal_normal
import torch
import torch.nn as nn

from .model_utils import sequence_mask, PackedDropout

class DiscriminativeVMD(nn.Module):
    pass

class VMDWithZHRec(nn.Module):
    def __init__(self, **kwargs):
        super(VMDWithZHRec, self).__init__()
        self.dropout_vmd_in = kwargs["dropout_vmd_in"]
        self.dropout_vmd = kwargs["dropout_vmd"]
        self.vmd_cell_type = kwargs["vmd_cell_type"]
        self.h_size = kwargs["h_size"]
        self.x_size = kwargs["x_size"]
        self.y_size = kwargs["y_size"]
        self.z_size = kwargs["z_size"]
        self.g_size = kwargs["g_size"]
        self.daily_att = kwargs["daily_att"]

        # layers
        self.vmd_in_dropout_layer = nn.Dropout(self.dropout_vmd_in)

        if self.vmd_cell_type == "ln-lstm":
            rnn = nn.LSTM
        else:
            rnn = nn.GRU
        self.rnn = nn.Sequential(
            rnn(
                input_size=self.x_size,
                hidden_size=self.h_size,
                batch_first=True,
                bidirectional=False,
            ),
            PackedDropout(self.dropout_vmd),
        )

        self.prior_h_linear = nn.Sequential(
            nn.Linear(in_features=self.x_size + self.h_size + self.z_size,
                out_features=self.z_size,
            ),
            nn.Tanh(),
        )
        self.post_h_linear = nn.Sequential(
            nn.Linear(in_features=self.x_size + self.h_size + self.y_size + self.z_size,
            out_features=self.z_size,
            ),
            nn.Tanh(),
        )
        self.prior_z_mean_linear = nn.Linear(self.z_size, self.z_size)
        self.prior_z_stddev_linear = nn.Linear(self.z_size, self.z_size)
        self.post_z_mean_linear = nn.Linear(self.z_size, self.z_size)
        self.post_z_stddev_linear = nn.Linear(self.z_size, self.z_size)
        
        self.g_linear = nn.Sequential(
            nn.Linear(self.h_size + self.z_size, self.g_size),
            nn.Tanh(),
        )

        self.y_linear = nn.Sequential(
            nn.Linear(self.g_size, self.y_size),
            nn.Softmax(dim=-1),
        )

        # functions
        self.packer = nn.utils.rnn.pack_padded_sequence
        self.padder = nn.utils.rnn.pad_packed_sequence
    
    def _z(self, h_z, mean_layer, stddev_layer, is_prior=False):
        mean = mean_layer(h_z)
        stddev = stddev_layer(h_z)
        stddev = torch.sqrt(torch.exp(stddev))
        epsilon = torch.randn(*h_z.shape, device=mean.device)

        z = mean if is_prior else mean + stddev * epsilon
        return z, (mean, stddev)

    def forward(self, x, y_true, n_days):
        """Forward pass for the Variational Movement Decoder (VMD)
        
        Parameters
        ----------
        x : tensor - (batch_size, max_valid_n_days, vmd_in_size)
            Input to VMD and output from MIE.
        y_true : tensor - (batch_size, max_valid_n_days, y_size)
            2-d vectorized movement (from input data)
        n_days : tensor - (batch_size,)
            Number of valid trading days in each sample

        Returns
        -------
        g_T: tensor - (batch_size, g_size)
            g for target day T
        y_T: tensor - (batch_size, y_size), optional
            predicted y for target day T
        kls: tensor - (batch_size, max_valid_n_days), optional (only in training phase).
            kl divergence between posterior and prior z distributions.
        """        
        device = x.device
        x = self.vmd_in_dropout_layer(x)
        max_valid_n_days = n_days.max()
        batch_size = x.shape[0]
        mask_aux_trading_days = sequence_mask(n_days - 1, max_valid_n_days)
        # print(f"vmd x = {x}")
        packed_batch = self.packer(x, n_days.to(device="cpu", dtype=torch.int64), 
            batch_first=True, enforce_sorted=False).to(device=x.device)
        rnn_out, _ = self.rnn(packed_batch)
        # print(f"vmd rnn_out = {rnn_out}")
        h_s = self.padder(rnn_out, batch_first=True, total_length=max_valid_n_days)[0]
        
        h_s = torch.transpose(h_s, 0, 1)
        x = torch.transpose(x, 0, 1) # max_n_days * batch_size * x_size
        y_ = torch.transpose(y_true, 0, 1) # max_n_days * batch_size * y_size

        z_shape = (max_valid_n_days, batch_size, self.z_size)
        z_priors = torch.zeros(*z_shape, device=device)
        z_posts = torch.zeros(*z_shape, device=device)
        kls = torch.zeros(*z_shape, device=device)
        z = torch.randn(batch_size, self.z_size, device=device)

        # print(f"""
        # h_s is on {h_s.device}
        # x is on {x.device}
        # y_ is on {y_.device}
        # z is on {z.device}
        # """)
        for i in range(max_valid_n_days):
            h_z_prior = self.prior_h_linear(torch.cat((x[i], h_s[i], z), dim=-1))
            z_prior, z_prior_pdf = self._z(
                h_z_prior,
                self.prior_z_mean_linear, self.prior_z_stddev_linear,
                is_prior=True,
            )

            h_z_post = self.post_h_linear(torch.cat((x[i], h_s[i], y_[i], z), -1))
            z_post, z_post_pdf = self._z(
                h_z_post, 
                self.post_z_mean_linear, self.post_z_stddev_linear,
                is_prior=False,
            )

            kl = kl_normal_normal(*z_post_pdf, *z_prior_pdf)

            z_priors[i] = z_prior
            z_posts[i] = z_post
            kls[i] = kl

            z = z_post
        
        h_s = h_s.transpose(0, 1)
        # print(f"vmd h_s = {h_s}")
        z_priors = z_priors.transpose(0, 1) # (batch_size, max_valid_n_days, z_size)
        z_posts = z_posts.transpose(0, 1) # (batch_size, max_valid_n_days, z_size)
        # print(f"vmd z_priors = {z_priors}")
        # print(f"vmd z_posts = {z_posts}")
        kls = kls.transpose(0, 1).sum(dim=2) # (batch_size, max_valid_n_days)

        g = self.g_linear(torch.cat((h_s, z_posts), -1)) # (batch_size, max_valid_n_days, g_size)
        # print(f"vmd g = {g}")
        y_pred = self.y_linear(g) # (batch_size, max_valid_n_days, y_size)

        sample_index = torch.arange(batch_size).unsqueeze(-1)
        day_index = torch.reshape(n_days - 1, (batch_size, 1))

        if self.training:
            g_T = g[sample_index, day_index].squeeze(1) # (batch_size, g_size)

            # print(f"vmd y_pred = {y_pred}")
            return_dict = {
                "g_T": g_T, "g": g, "y_pred": y_pred,
                "mask_aux_trading_days": mask_aux_trading_days,
                "kls": kls, "sample_index": sample_index, "day_index": day_index,
            }
            if not self.daily_att:
                y_T = y_pred[sample_index, day_index].squeeze(1) # (batch_size, g_size)
                return_dict["y_T"] = y_T
            return return_dict
        else:
            # evaluation. use priors for g & y.
            z_prior_T = z_priors[sample_index, day_index].squeeze(1)
            h_s_T = h_s[sample_index, day_index].squeeze(1)

            g_T = self.g_linear(torch.cat((h_s_T, z_prior_T), -1))

            return_dict = {
                "g_T": g_T, "g": g, "y_pred": y_pred, 
                "mask_aux_trading_days": mask_aux_trading_days,
                "kls": kls, "sample_index": sample_index, "day_index": day_index,
            }
            if not self.daily_att:
                y_T = self.y_linear(g_T)
                return_dict["y_T"] = y_T
            return return_dict

class VMDWithHRec(nn.Module):
    pass

class VMD(nn.Module):
    def __init__(self, **kwargs):
        super(VMD, self).__init__()
        self.variant_type = kwargs["variant_type"]
        self.vmd_rec = kwargs["vmd_rec"]

        if self.variant_type == 'discriminative':
            # vmd = DiscriminativeVMD()
            pass
        else:
            if self.vmd_rec == "h":
                # vmd = VMDWithHRec()
                pass
            else:
                vmd = VMDWithZHRec(**kwargs)
        self.vmd = vmd


    def forward(self, *args, **kwargs):
        return self.vmd(*args, **kwargs)






