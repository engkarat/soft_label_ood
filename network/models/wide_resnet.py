import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        ) if not self.equalInOut else None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = out if self.equalInOut else x
        out = self.conv1(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        sh_cut = self.convShortcut(x) if not self.equalInOut else x
        return torch.add(sh_cut, out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(
        self, depth, num_classes, widen_factor=1, drop_rate=0.0, input_n_channel=3,
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            input_n_channel, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.drop_rate = drop_rate
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, act_drop=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.drop_rate:
            out = F.dropout(out, p=self.drop_rate, training=act_drop if act_drop else self.training)
        out = self.fc(out)
        return out


def wide_resnet40_4(n_classes, drop_rate=0.0):
    return WideResNet(40, n_classes, widen_factor=4, drop_rate=drop_rate)

def wide_resnet28_10(n_classes, drop_rate=0.0):
    return WideResNet(28, n_classes, widen_factor=10, drop_rate=drop_rate)
