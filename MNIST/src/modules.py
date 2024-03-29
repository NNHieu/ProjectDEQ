import imp
import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

from src.datamodules import get_normalize_layer

from deq.standard.models import core as std_deq
from deq.mon import mon, splitting as sp
from deq.mon.train import expand_args, MON_DEFAULTS

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8, init_std=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        # norm_layer = nn.BatchNorm2d
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)

        self.conv1.weight.data.normal_(0, init_std)
        self.conv2.weight.data.normal_(0, init_std)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

class Net(nn.Module):
    def __init__(self, core, in_trans, out_trans):
        super(Net, self).__init__()
        self.normalize_input = get_normalize_layer()
        self.in_trans = in_trans
        self.core = core
        self.out_trans = out_trans

    def forward(self, x, **kwargs):
        x_norm = self.normalize_input(x)
        phi_x = self.in_trans(x_norm)
        z, jac_loss, sradius = self.core(phi_x, **kwargs)
        out = self.out_trans(z)
        return out, jac_loss, sradius

def get_model(arch, init_std=0.01):
    in_trans = nn.Sequential(
        nn.Conv2d(1, arch.h_features[0], kernel_size=3, bias=True, padding=1),
        # nn.GroupNorm(4, arch.h_features[0]),
        nn.BatchNorm2d(arch.h_features[0]),

    )
    f = ResNetLayer(arch.h_features[0], arch.h_features[1], kernel_size=3, init_std=init_std)
    if arch.get('core', 'deq') == 'deq':
        core = std_deq.DEQLayer(f, arch)
    elif arch.core == 'recur':
        core = std_deq.RecurLayer(f, arch.num_layers)
    else:
        raise NotImplemented
    out_trans = nn.Sequential(
                            nn.BatchNorm2d(arch.h_features[0]),
                            nn.AvgPool2d(7,7),
                            nn.Flatten(),
                            nn.Linear(arch.h_features[0]*4*4,10))
    return Net(core, in_trans, out_trans)
