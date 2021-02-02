import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):

    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, growth_rate, block, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, drop_rate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet(nn.Module):

    def __init__(
        self, depth, num_classes, growth_rate=12, reduction=0.5,
        bottleneck=True, drop_rate=0.0, input_n_channel=3,
    ):
        super(DenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4)//3
        if bottleneck == True:
            n = n//2
            block = BottleneckBlock
        else:
            block = BasicBlock
        self.drop_rate = drop_rate
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(
            input_n_channel, in_planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, drop_rate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)

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
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.drop_rate:
            out = F.dropout(out, p=self.drop_rate, training=act_drop if act_drop else self.training)
        out = self.fc(out)
        return out


def densenet100_bc(n_classes, drop_rate=0.):
    return DenseNet(100, n_classes, bottleneck=True, drop_rate=drop_rate)
