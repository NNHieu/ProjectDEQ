""" recurrent_net.py
    Parity solving recurrent convolutional neural network.
    April 2021
"""
import os
import sys

DEQ_LIB = os.path.abspath(os.path.join(sys.path[0], "../src"))
if DEQ_LIB not in sys.path:
    sys.path.append(DEQ_LIB)
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

import deq

from .generic import BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class RecurNet(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, depth, block_args):
        super(RecurNet, self).__init__()
        assert (depth - 4) % 4 == 0, "Depth not compatible with recurrent architecture."
        self.iters = (depth - 4) // 4
        self.in_planes = int(width)
        self.conv1 = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1, bias=False)
        layers = []
        for i in range(len(num_blocks)):
            layers.append(
                self._make_layer(block, width, num_blocks[i], stride=1, block_args=block_args)
            )

        self.recur_block = nn.Sequential(*layers)
        self.recur_layer = deq.core.RecurLayer(self.recur_block, self.iters)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(
            width, int(width / 2), kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv4 = nn.Conv1d(int(width / 2), 2, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride, block_args):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, **block_args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        if self.training:
            self.thoughts = None

            out = F.relu(self.conv1(x))
            out = self.recur_layer(out, self.iters)
            thought = self._project_thought(out)
        else:
            self.thoughts = torch.zeros((self.iters, x.size(0), 2, x.size(2))).to(x.device)

            out = F.relu(self.conv1(x))
            # self.thoughts = torch.zeros((self.iters, out.size(0), out.size(1), out.size(2))).to(x.device)
            # self.recur_layer(out, self.iters, proj_out = lambda i, y: self._save_thought(self.thoughts, i, y))
            out = self.recur_layer(
                out,
                self.iters,
                proj_out=lambda i, y: self._save_thought(
                    self.thoughts, i, self._project_thought(y)
                ),
            )
            thought = self.thoughts[-1]

        return thought

    def _project_thought(self, out):
        thought = F.relu(self.conv2(out))
        thought = F.relu(self.conv3(thought))
        return self.conv4(thought)

    @staticmethod
    def _save_thought(thoughts, i, thought):
        thoughts[i] = thought


def recur_net(depth, width, block_args, **kwargs):
    return RecurNet(BasicBlock, [2], width, depth, block_args)
