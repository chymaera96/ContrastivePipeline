import unittest

import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


import util
from ntxent import ntxent_loss
from sfnet.gpu_transformations import GPUTransformNeuralfp
from sfnet.data_sans_transforms import NeuralfpDataset
from sfnet.modules.simclr import SimCLR
from sfnet.modules.moco import MoCo
from sfnet.modules.residual import SlowFastNetwork, ResidualUnit
from baseline.encoder import Encoder
from baseline.neuralfp import Neuralfp
from eval import eval_faiss
from test_fp import create_fp_db, create_dummy_db

class MoCoEncoderTestCase(unittest.TestCase):
    def setUp(self):
        self.cfg = util.load_config('config/config3.yaml')
        self.base_encoder = SlowFastNetwork(ResidualUnit, self.cfg)
        self.model = MoCo(self.cfg, self.base_encoder)
        self.device = torch.device("cuda")
        self.x_i = torch.randn(2, 1, 1000).to(self.device)
        self.x_j = torch.randn(2, 1, 1000).to(self.device)

    def test_moco_storage_address(self):
        # Get the first tensor in x and y
        x = next(iter(self.model.encoder_q.parameters()))
        y = next(iter(self.model.encoder_k.parameters()))

        x_ptrs = set(e.data_ptr() for e in x.view(-1))
        y_ptrs = set(e.data_ptr() for e in y.view(-1))
        flag = (x_ptrs < y_ptrs) or (y_ptrs < x_ptrs)
        self.assertTrue(not flag, 
                        'The addresses of the two encoders are the same')


    def test_moco_pointer_address(self):
        for param_q, param_k in zip(
            self.model.encoder_q.parameters(), self.model.encoder_k.parameters()
        ):
            self.assertTrue(param_q.data_ptr() != param_k.data_ptr(), 
                            'The pointers of the two encoders are the same')
            break

    def test_moco_pointer_address2(self):
        for param_q, param_k in zip(
            self.model.encoder_q.parameters(), self.model.encoder_k.parameters()
        ):
            self.assertTrue(param_q is not param_k, 
                            'The pointers of the two encoders are the same')
            break


if __name__ == '__main__':
    unittest.main()