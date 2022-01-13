import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

DEQ_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
if DEQ_LIB not in sys.path:
    sys.path.append(DEQ_LIB)
import deq



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
            self.sigma = F.tanh

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
    def __init__(self, in_feature, h_feature, out_feature, in_trans, f, deq_conf):
        super(Net, self).__init__()
        self.in_trans = in_trans
        self.core = deq.core.DEQLayer(f, deq_conf)
        self.out_trans = nn.Linear(h_feature, out_feature)

    def forward(self, x, **kwargs):
        phi_x = self.in_trans(x)
        z, jac_loss, sradius = self.core(phi_x, **kwargs)
        out = self.out_trans(z)
        return out, jac_loss, sradius

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
    f = Block(arch.h_features, activation=arch.block.activation)
    f.apply(lambda m: init_weights(m, std=init_std))
    return Net(arch.in_features, arch.h_features, arch.out_features, in_trans, f, arch)
