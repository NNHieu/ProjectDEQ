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

from .generic import BasicBlock, BasicInjectedBlock, InjectedBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


def small_init_weights(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0.0, 0.01)


class FixedPointNet(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, conf):
        super(FixedPointNet, self).__init__()
        self.in_planes = int(width)
        # In
        self.conv1 = nn.Conv1d(1, width, kernel_size=3, stride=1, padding=1, bias=False)
        # layers = []
        # for i in range(len(num_blocks)):
        #     layers.append()
        # Core
        self.recur_block = InjectedBlock(block, self.in_planes, width, num_blocks[0], stride=1)
        self.fixedpoint_layer = deq.core.DEQLayer(self.recur_block, conf)
        # Out
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(
            width, int(width / 2), kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv4 = nn.Conv1d(int(width / 2), 2, kernel_size=3, stride=1, padding=1, bias=False)
        if conf.get("small_init"):
            self.recur_block.apply(small_init_weights)

    def forward(self, x, deq_mode=True):
        if self.training:
            self.thoughts = None

            out = F.relu(self.conv1(x))
            out = self.fixedpoint_layer(out, deq_mode=deq_mode)
            thought = self._project_thought(out)
        else:
            self.thoughts = None

            out = F.relu(self.conv1(x))
            out = self.fixedpoint_layer(out, deq_mode=deq_mode)
            thought = self._project_thought(out)

        return thought

    def _project_thought(self, out):
        thought = F.relu(self.conv2(out))
        thought = F.relu(self.conv3(thought))
        return self.conv4(thought)

    @staticmethod
    def _save_thought(thoughts, i, thought):
        thoughts[i] = thought


def fp_net(cfg, width, **kwargs):
    return FixedPointNet(BasicInjectedBlock, [2], width, cfg)
