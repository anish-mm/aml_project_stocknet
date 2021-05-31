import numpy as np
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__()

        self.g_size = kwargs["g_size"]
        self.y_size = kwargs["y_size"]
        self.daily_att = kwargs["daily_att"]
        self.att_c_size = self.y_size if self.daily_att == 'y' else self.g_size

        # layers
        self.proj_i_linear = nn.Sequential(
            nn.Linear(self.g_size, self.g_size, bias=False),
            nn.Tanh(),
        )
        self.w_i_linear = nn.Linear(self.g_size, 1)
        self.proj_d_linear = nn.Sequential(
            nn.Linear(self.g_size, self.g_size, bias=False),
            nn.Tanh(),
        )
        self.aux_soft = nn.Softmax(dim=-1)
        self.y_linear = nn.Sequential(
            nn.Linear(self.att_c_size + self.g_size, self.y_size),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, g, g_T, mask_aux_trading_days, y_pred=None):
        """Temporal attention to get final y_T prediction

        Parameters
        ----------
        g : tensor - (batch_size, max_valid_n_days, g_size)
            g matrix
        g_T : tensor - (batch_size, g_size)
            g for the target day for each sample
        mask_aux_trading_days : bool tensor - (batch_size, max_valid_n_days)
            mask for all auxiliary trading days (i.e. excluding the target day)
        y_pred : tensor - (batch_size, max_valid_n_days, y_size), optional
            predicted movements, by default None

        Returns
        -------
        y_T : tensor - (batch_size, y_size)
            predicted movements for target day.
        v_star : tensor - (batch_size, max_valid_n_days)
        """        
        if self.daily_att == 'y':
            assert y_pred is not None, "y cannot be None when daily_att = y."

        # information score
        proj_i = self.proj_i_linear(g)
        v_i = self.w_i_linear(proj_i).squeeze(-1) # (batch_size, max_valid_n_days)

        # dependency score
        proj_d = self.proj_d_linear(g) # (batch_size, max_valid_n_days, g_size)
        v_d = torch.matmul(proj_d, g_T.unsqueeze(-1)).squeeze(-1) # (batch_size, max_valid_n_days)
        # (b, d, g) x (b, g, 1) -> (b, d, 1) -> squeeze -> (b, d)

        aux_score = v_i * v_d
        # print(f"aux_score={aux_score}")
        masked_aux_score = torch.where(mask_aux_trading_days, aux_score, torch.tensor(-np.inf, device=aux_score.device))
        # print(f"masked_aux_score={masked_aux_score}")
        v_star = self.aux_soft(masked_aux_score)

        if self.daily_att == 'y':
            context = y_pred.transpose(1, 2) # (batch_size, y_size, max_valid_n_days)
        else:
            context = g.transpose(1, 2) # (batch_size, g_size, max_valid_n_days)
        v_star = v_star.unsqueeze(-1) # (batch_size, max_valid_n_days, 1)
        # print(f"context = {context}")
        # print(f"v_star={v_star}")
        att_c = torch.matmul(context, v_star).squeeze(-1) # (batch_size, g_size | y_size)
        # print(f"att_c={att_c}")
        y_T = self.y_linear(torch.cat((att_c, g_T), -1))

        return y_T, v_star.squeeze(-1)






