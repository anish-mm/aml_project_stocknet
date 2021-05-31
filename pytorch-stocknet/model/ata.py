import torch
import torch.nn as nn

class GenerativeATA(nn.Module):
    def __init__(self, **kwargs):
        super(GenerativeATA, self).__init__()
        self.alpha = kwargs["alpha"]
        self.use_constant_kl_lambda = kwargs["use_constant_kl_lambda"]
        self.constant_kl_lambda = kwargs["constant_kl_lambda"]
        self.kl_lambda_anneal_rate = kwargs["kl_lambda_anneal_rate"]
        self.kl_lambda_start_step = kwargs["kl_lambda_start_step"]

    def _kl_lambda(self, global_step):
        if global_step < self.kl_lambda_start_step:
            return 0.0
        elif self.use_constant_kl_lambda:
            return self.constant_kl_lambda
        else:
            return min(self.kl_lambda_anneal_rate * global_step, 1.0)

    def forward(self, v_star, y_true, y_pred, y_T,
        kls, global_step, sample_index, day_index
        ):
        """Compute objective function for the batch.

        Parameters
        ----------
        v_star : tensor - (batch_size, max_valid_n_days)
            temporal attention weights
        y_true : tensor - (batch_size, max_valid_n_days, price_size)
            true price (from input data)
        y_pred : tensor - (batch_size, max_valid_n_days, price_size)
            predicted price
        y_T : tensor - (batch_size, price_size)
            price prediction for target day
        kls : tensor - (batch_size, max_valid_n_days)
            kl-divergences between posterior and prior z distributions
        global_step : scalar
            number of batches trained on till this point
        sample_index : tensor - (batch_size, 1)
            used for indexing to each sample
        day_index : tensor - (batch_size, 1)
            used for indexing to last time step of each sample.

        Returns
        -------
        Loss : tensor - ()
            mean loss for the batch.
        """        
        batch_size = y_true.shape[0]

        v_aux = self.alpha * v_star # (batch_size, max_valid_n_days)
        # print(f"v_aux.shape={v_aux.shape}")
        # print(f"alpha={self.alpha}")
        # print(f"v_star.shape={v_star.shape}")

        minor = 0.0 # 0.0, 1e-7
        # print(f"y_pred = {y_pred}")
        likelihood_aux = torch.sum(y_true * torch.log(y_pred + minor), dim=2) # (batch_size, max_valid_n_days)

        kl_lambda = self._kl_lambda(global_step)
        obj_aux = likelihood_aux - kl_lambda * kls # (batch_size, max_valid_n_days)
        # print(f"obj_aux = {obj_aux}")
        # deal with T specially, likelihood_T: batch_size, 1
        y_T_ = y_true[sample_index, day_index].squeeze(1) # batch_size, y_size
        # print(f"y_T_ = {y_T_}")
        likelihood_T = torch.sum(y_T_ * torch.log(y_T + minor), dim=1, keepdim=True)
        # print(f"y_T = {y_T}")
        # print(f"likelihood_T = {likelihood_T}")
        kl_T = kls[sample_index, day_index].reshape(batch_size, 1)
        # print(f"kl_T = {kl_T}")
        obj_T = likelihood_T - kl_lambda * kl_T
        # print(f"obj_T = {obj_T}")
        # print(f"obj_T.shape={obj_T.shape}")
        # print(f"obj_aux.shape={obj_aux.shape}, v_aux.shape={v_aux.shape}")
        term2=torch.sum(obj_aux * v_aux, axis=1, keepdim=True)
        # print(f"term2 = {term2}")
        # print(f"term2.shape={term2.shape}")
        obj = obj_T + term2 # (batch_size, 1)
        # print(f"obj = {obj}")
        loss = torch.mean(-obj)

        return loss

class ATA(nn.Module):
    def __init__(self, **kwargs):
        super(ATA, self).__init__()

        self.variant_type = kwargs["variant_type"]

        if self.variant_type == "discriminative":
            # ata = DiscriminativeATA
            pass
        else:
            ata = GenerativeATA
        self.ata = ata(**kwargs)
    
    def forward(self, *args, **kwargs):
        return self.ata(*args, **kwargs)
