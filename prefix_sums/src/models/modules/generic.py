import torch
from torch._C import Argument
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    """Basic residual block class"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type="id"):
        super(BasicBlock, self).__init__()
        if norm_type == "id":
            norm_layer = nn.Identity
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError(f"{norm_type} is not supported")

        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = norm_layer(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                )
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.norm1(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.norm2(out)
        return out


class BasicInjectedBlock(nn.Module):
    """Basic residual block class"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type="batch"):
        super(BasicInjectedBlock, self).__init__()
        if norm_type == "id":
            norm_layer = nn.Identity
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError(f"{norm_type} is not supported")
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = norm_layer(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                )
            )
        self.norm3 = norm_layer(planes)

    def forward(self, z, x):
        out = self.norm1(F.relu(self.conv1(z)))
        out = self.conv2(out)
        out = self.norm2(out + x)
        out += self.shortcut(z)
        out = self.norm3(F.relu(out))
        return out


class InjectedBlock(nn.Module):
    def __init__(self, block, in_planes, planes, num_blocks, stride, **bloc_args):
        super(InjectedBlock, self).__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        self.layers = []
        for strd in strides:
            self.layers.append(block(in_planes, planes, strd, **bloc_args))
            in_planes = planes * block.expansion
        self.layers = nn.Sequential(*self.layers)

    def forward(self, z, x):
        for layer in self.layers:
            z = layer(z, x)
        return z
