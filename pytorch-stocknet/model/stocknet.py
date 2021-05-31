"""
The fully assembled stocknet model.
"""
import torch.nn as nn
import sys

from .compacting_mie import MIE
from .vmd import VMD
from .temporal_attention import TemporalAttention
from .ata import ATA

class Stocknet(nn.Module):
    def __init__(self, **kwargs):
        super(Stocknet, self).__init__()

        self.daily_att = kwargs["daily_att"]

        # parts
        self.mie = MIE(**kwargs)

        self.vmd = VMD(**kwargs, x_size=self.mie.x_size)
        if self.daily_att:
            self.temporal_attention = TemporalAttention(**kwargs)
        self.ata = ATA(**kwargs)
    
    def forward(self, **kwargs):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """        
        x, max_n_days = self.mie(
            prices=kwargs["prices"], words=kwargs["words"],
            n_days=kwargs["n_days"], n_msgs=kwargs["n_msgs"],
            n_words=kwargs["n_words"], ss_indices=kwargs["ss_indices"],
        )

        y_true=kwargs["y_true"][:, :max_n_days]
        vmd_out = self.vmd(
            x=x, y_true=y_true,
            n_days=kwargs["n_days"]
        )

        if self.daily_att:
            y_T, v_star = self.temporal_attention(
               g=vmd_out["g"], g_T=vmd_out["g_T"],
               mask_aux_trading_days=vmd_out["mask_aux_trading_days"],
               y_pred=vmd_out["y_pred"],
            )

            if self.training: # is this required?
                loss = self.ata(
                v_star=v_star, y_true=y_true, y_pred=vmd_out["y_pred"],
                y_T=y_T, kls=vmd_out["kls"],
                sample_index=vmd_out["sample_index"], day_index=vmd_out["day_index"],
                global_step=kwargs["global_step"],
                )
                return y_T, loss
            else:
                return y_T
        else:
            y_T = vmd_out["y_T"]
            s = """
            In this case, we are not using the ATA. I think this is same as the 
            INDEPENDENTANALYST scenario, and this is some part of the code that
            they didn't clean up. The correct way to get the INDEPENDENTANALYST 
            in the latest code is to set alpha=0 in the configs.
            One could still proceed with this, and pass alpha=0 and 
            v_star=<some tensor - (batch_size, max_valid_n_days)> to accomplish 
            this. Left open here since it is not required.
            """
            print(s)
            sys.exit(1)
        
        



