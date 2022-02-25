import imp
import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

from deq.standard.models import core as std_deq
from deq.mon import mon, splitting as sp
from deq.mon.train import expand_args, MON_DEFAULTS

class SingleMonDEQFcLayer(nn.Module):

    def __init__(self, splittingMethod, in_dim, out_dim, m=0.1, **kwargs):
        super().__init__()
        linear_module = mon.MONSingleFc(in_dim, out_dim, m=m)
        nonlin_module = mon.MONReLU()
        self.mon = splittingMethod(linear_module, nonlin_module, **expand_args(MON_DEFAULTS, kwargs))
        self.stats = self.mon.stats

    def forward(self, x, **kwargs):
        x = x.view(x.shape[0], -1)
        z = self.mon(x)
        return z[-1]