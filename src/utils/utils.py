import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DoubleTanh(nn.Module):
    def __init__(self, _lambda):
        super(DoubleTanh, self).__init__()
        self._lambda = _lambda

    def forward(self, P, R):
        m = -0.5 * F.tanh(self._lambda*(P-2*R)) + 0.5 * F.tanh(self._lambda*(P-2*(1-R)))
        return m