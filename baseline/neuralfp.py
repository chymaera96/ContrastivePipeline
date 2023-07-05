import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

d = 128
h = 1024
u = 32
v = int(h/d)
class Neuralfp(nn.Module):
    def __init__(self, encoder):
        super(Neuralfp, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Conv1d(h, d * u, kernel_size=(1,), groups=d),
                                        nn.ELU(),
                                        nn.Conv1d(d * u, d, kernel_size=(1,), groups=d)
                                        )


    def forward(self, x_i, x_j):
        
        h_i = self.encoder(x_i)
        z_i = self.projector(h_i.unsqueeze(-1)).squeeze(-1)
        z_i = F.normalize(z_i, p=2)

        h_j = self.encoder(x_j)
        z_j = self.projector(h_j.unsqueeze(-1)).squeeze(-1)
        z_j = F.normalize(z_j, p=2)

        return h_i, h_j, z_i, z_j