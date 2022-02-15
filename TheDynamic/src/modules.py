import imp
import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

from deq.standard.models import core as std_deq
from deq.mon import mon, splitting as sp
from deq.mon.train import expand_args, MON_DEFAULTS

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

class Block(nn.Module):
    def __init__(self, channel, activation="relu") -> None:
        super(Block, self).__init__()
        norm_layer = nn.Identity
        self.norm1 = norm_layer(channel)
        self.norm2 = norm_layer(channel)

        self.lin1 = nn.Linear(channel, channel, bias=False)
        self.lin2 = nn.Linear(channel, channel, bias=False)

        if activation == 'relu':
            self.sigma=F.relu
        elif activation == 'tanh':
            self.sigma = torch.tanh

    def forward(self, z, x):
        out = self.norm1(self.sigma(self.lin1(z)))
        out = self.lin2(out) + x
        out = self.norm2(out)
        return self.sigma(out)


class PaddingBlock(nn.Module):
    def __init__(self, channel) -> None:
        super(PaddingBlock, self).__init__()
        self.channel = channel

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.channel, device=x.device)
        out[:, : x.shape[1]] += x
        return out


class AxisProject(nn.Module):
    def __init__(self, channel) -> None:
        super(AxisProject, self).__init__()
        self.channel = channel

    def forward(self, x):
        return x[:, : self.channel]


class Net(nn.Module):
    def __init__(self, core, h_feature, out_feature, in_trans):
        super(Net, self).__init__()
        self.in_trans = in_trans
        self.core = core
        self.out_trans = nn.Linear(h_feature, out_feature)

    def forward(self, x, **kwargs):
        phi_x = self.in_trans(x)
        z, jac_loss, sradius = self.core(phi_x, **kwargs)
        out = self.out_trans(z)
        return out, jac_loss, sradius

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
        return z[-1], None, None

def init_weights(m, std=1.0):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, std)

def get_model(arch, init_std=1.0):
    if arch.in_trans == "linear":
        in_trans = nn.Sequential(
            nn.Linear(arch.in_features, arch.h_features), nn.ReLU(inplace=True)
        )
    elif arch.in_trans == "padding":
        in_trans = PaddingBlock(arch.h_features)
        # arch.get('core', 'deq')
    core_type = arch.get('core', 'deq')
    if core_type == 'deq':
        f = Block(arch.h_features, activation=arch.block.activation)
        f.apply(lambda m: init_weights(m, std=init_std))
        core = std_deq.DEQLayer(f, arch)
    elif core_type == 'single_step':
        f = Block(arch.h_features, activation=arch.block.activation)
        core = std_deq.RecurLayer(f, arch.f_thres)
    elif core_type == 'mondeq':
        core = SingleMonDEQFcLayer(sp.MONPeacemanRachford, arch.h_features, arch.h_features, alpha=1.0,
                    max_iter=300,
                    tol=1e-3,
                    m=1.0)
    return Net(core, arch.h_features, arch.out_features, in_trans)
