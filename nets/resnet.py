'''
ResNet-v2 is desired.
Fix bugs for ResNet.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

cfg = {
    18: [2, 2, 2, 2],
    20: [3, 3, 3],
    32: [5, 5, 5],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    56: [9, 9, 9],
    110: [18, 18, 18],
    164: [27, 27, 27]
}

def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, strides=2):
        super(ResidualBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if strides != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=strides, bias=False)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn0(x))
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x += shortcut
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, strides=2):
        # in_planes: actual channel num of input
        # out_planes: intermediate channel num
        super(Bottleneck, self).__init__()
        expansion = 4
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes*expansion, kernel_size=1, stride=1, bias=False)
        self.shortcut = None
        if strides != 1 or in_planes != out_planes*expansion:
            self.shortcut = nn.Conv2d(in_planes, out_planes*expansion, kernel_size=1, stride=strides, bias=False)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn0(x))
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x += shortcut
        return x

class ResNet(nn.Module):
    def __init__(self, n_layer, n_class):
        super(ResNet, self).__init__()

        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.n_class = n_class
        self.block_n_cell = cfg[n_layer]
        self.imagenet = len(self.block_n_cell) > 3

        if self.imagenet:
            self.base_n_channel = 64
            self.cell_fn = Bottleneck if n_layer >= 50 else ResidualBlock
            self.conv0 = nn.Conv2d(3, self.base_n_channel, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.base_n_channel = 16
            self.cell_fn = ResidualBlock
            self.conv0 = nn.Conv2d(3, self.base_n_channel, 3, stride=1, padding=1, bias=False)

        self.expansion = 4 if self.cell_fn is Bottleneck else 1
        self.block_list = self._block_layers()
        self.bn = nn.BatchNorm2d(self.base_n_channel*(2**(len(self.block_n_cell)-1)) * self.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.base_n_channel*(2**(len(self.block_n_cell)-1)) * self.expansion, self.n_class)
        self.apply(_weights_init)

    def _block_fn(self, in_planes, out_planes, n_cell, strides):
        # the 1-st layer in 1-st block
        if strides == 2:
            in_planes *= self.expansion
        blocks = [self.cell_fn(in_planes, out_planes, strides)]
        for _ in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes*self.expansion, out_planes, 1))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i == 0:
                block_list.append(self._block_fn(self.base_n_channel, self.base_n_channel, n_cell, 1))
            else:
                block_list.append(self._block_fn(self.base_n_channel*(2**(i-1)), self.base_n_channel*(2**i), n_cell, 2))
        return nn.ModuleList(block_list)

    def forward(self, x):
        x = self.conv0(x)
        if self.imagenet:
            x = self.maxpool(x)
        for blocks in self.block_list:
            for block in blocks[:]:
                x = block(x)
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = ResNet(20, 100)
    x = torch.zeros([1, 3, 32, 32])
    y = net(x)
    print(net)

    # macs, params = profile(net, inputs=(x,))
    # print(macs, params)
