""" recurrent_net.py
    Parity solving recurrent convolutional neural network.
    April 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from .generic import BasicInjectedBlock, InjectedBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class RecurInjectedNet(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, depth):
        super(RecurInjectedNet, self).__init__()
        assert (depth - 4) % 4 == 0, "Depth not compatible with recurrent architecture."
        self.iters = (depth - 4) // 4
        self.in_planes = int(width)
        self.conv1 = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1, bias=False)

        self.recur_block = InjectedBlock(block, self.in_planes, width, num_blocks[0], stride=1)
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

    def forward(self, x, **kwargs):
        if self.training:
            self.thoughts = None

            new_x = F.relu(self.conv1(x))
            out = torch.zeros_like(new_x)
            for i in range(self.iters):
                out = self.recur_block(out, new_x)
            thought = self._project_thought(out)
        else:
            self.thoughts = torch.zeros((self.iters, x.size(0), 2, x.size(2))).to(x.device)

            new_x = F.relu(self.conv1(x))
            out = torch.zeros_like(new_x)
            for i in range(self.iters):
                out = self.recur_block(out, new_x)
                self.thoughts[i] = self._project_thought(out)
            thought = self.thoughts[-1]

        return thought

    def _project_thought(self, out):
        thought = F.relu(self.conv2(out))
        thought = F.relu(self.conv3(thought))
        return self.conv4(thought)


def recur_injected_net(depth, width, **kwargs):
    return RecurInjectedNet(BasicInjectedBlock, [2], width, depth)
