""" feed_forward_net.py
    Prefix sum solving convolutional neural network.
    May 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from .generic import BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class FFNet(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, depth):
        super(FFNet, self).__init__()
        assert (depth - 4) % 4 == 0, "Depth not compatible with recurrent architectue."
        self.iters = (depth - 4) // 4
        self.in_planes = int(width)
        self.conv1 = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for _ in range(self.iters):
            for i in range(len(num_blocks)):
                layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        self.recur_block = nn.Sequential(*layers)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(
            width, int(width / 2), kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv4 = nn.Conv1d(int(width / 2), 2, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.recur_block(out)
        thought = F.relu(self.conv2(out))
        thought = F.relu(self.conv3(thought))
        thought = self.conv4(thought)
        return thought


def ff_net(depth, width, **kwargs):
    return FFNet(BasicBlock, [2], width, depth)
