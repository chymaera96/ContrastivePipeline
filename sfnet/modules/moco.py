import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    def __init__(self, cfg, base_encoder):
        super(MoCo, self).__init__()
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder
        d = cfg['dim']
        h = cfg['h']
        u = cfg['u']
        m = cfg['momentum']
        self.m = m

        self.projector = nn.Sequential(nn.Conv1d(h, d * u, kernel_size=(1,), groups=d),
                                        nn.ELU(),
                                        nn.Conv1d(d * u, d, kernel_size=(1,), groups=d)
                                        )
        

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not updated by gradient
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x_i, x_j):
        
        h_i = self.encoder_q(x_i)
        z_i = self.projector(h_i.unsqueeze(-1)).squeeze(-1)
        z_i = F.normalize(z_i, p=2)

        with torch.no_grad():
            self._momentum_update_key_encoder()

        h_j = self.encoder_k(x_j)
        z_j = self.projector(h_j.unsqueeze(-1)).squeeze(-1)
        z_j = F.normalize(z_j, p=2)



        return h_i, h_j, z_i, z_j