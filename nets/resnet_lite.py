import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet import _weights_init
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

_BATCH_NORM_DECAY = 0.1
_EPSILON = 1e-5

def conv_flops(inputs, outputs, kernel_size):
    _, c_in, h_in, w_in = inputs.size()
    _, c_out, h_out, w_out = outputs.size()
    return kernel_size*kernel_size*c_in*c_out*h_out*w_out

def fc_flops(inputs, outputs):
    _, c_in = inputs.size()
    _, c_out = outputs.size()
    return c_in*c_out

class ResidualBlockLite(nn.Module):
    '''
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    '''
    def __init__(self, in_planes, out_planes_list, stride=2, project=False):
        super(ResidualBlockLite, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        self.bn0 = nn.BatchNorm2d(in_planes, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv1 = nn.Conv2d(in_planes, out_planes_1, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes_1, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.shortcut = None
        # if stride != 1 and len(out_planes_list) > 2:
        if project:
            self.shortcut = nn.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride,
                                      padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=1, padding=1, bias=False)

    def cnt_flops(self, x):
        cnt_flops = 0
        x = F.relu(self.bn0(x))
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            cnt_flops += conv_flops(x, shortcut, 1)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        cnt_flops += conv_flops(x, conv1, 3)
        conv2 = self.conv2(conv1)
        cnt_flops += conv_flops(conv1, conv2, 3)
        return cnt_flops

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn0(x))
        if self.shortcut != None:
            shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x += shortcut
        return x

class BottleneckLite(nn.Module):
    '''
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c3, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    '''
    def __init__(self, in_planes, out_planes_list, stride=2, project=False):
        super(BottleneckLite, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        out_planes_3 = out_planes_list[2]
        self.bn0 = nn.BatchNorm2d(in_planes, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv1 = nn.Conv2d(in_planes, out_planes_1, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes_1, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv2 = nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes_2, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.shortcut = None
        # if stride != 1 and len(out_planes_list) > 2:
        if project:
            self.shortcut = nn.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride,
                                      padding=0, bias=False)
        self.conv3 = nn.Conv2d(out_planes_2, out_planes_3, kernel_size=1, stride=1, bias=False)

    def cnt_flops(self, x):
        cnt_flops = 0
        x = F.relu(self.bn0(x))
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            cnt_flops += conv_flops(x, shortcut, 1)
        conv1 = F.relu(self.bn1(self.conv1(x)))
        cnt_flops += conv_flops(x, conv1, 1)
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        cnt_flops += conv_flops(conv1, conv2, 3)
        conv3 = self.conv3(conv2)
        cnt_flops += conv_flops(conv2, conv3, 1)
        return cnt_flops

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


class ResNetL(nn.Module):
    def __init__(self, n_layer, n_class, channel_lists):
        super(ResNetL, self).__init__()

        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.n_class = n_class
        self.channel_lists = channel_lists
        self.block_n_cell = cfg[n_layer]
        self.base_n_channel = channel_lists[0]
        self.imagenet = len(self.block_n_cell) > 3

        if self.imagenet:
            self.cell_fn = BottleneckLite if n_layer >= 50 else ResidualBlockLite
            self.conv0 = nn.Conv2d(3, self.base_n_channel, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.cell_fn = ResidualBlockLite
            self.conv0 = nn.Conv2d(3, self.base_n_channel, 3, stride=1, padding=1, bias=False)

        self.block_list = self._block_layers()
        self.bn = nn.BatchNorm2d(channel_lists[-1][-1][-1], momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_lists[-1][-1][-1], self.n_class)
        self.apply(_weights_init)

    def _block_fn(self, in_planes, out_planes_lists, n_cell, strides):
        blocks = [self.cell_fn(in_planes, out_planes_lists[0], strides, project=True)]
        for i in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes_lists[i-1][-1], out_planes_lists[i], 1))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i == 0:
                block_list.append(self._block_fn(self.base_n_channel, self.channel_lists[i+1], n_cell, 1))
            else:
                in_planes = self.channel_lists[i][-1][-1]
                block_list.append(self._block_fn(in_planes, self.channel_lists[i+1], n_cell, 2))
        return nn.ModuleList(block_list)

    def cnt_flops(self, x):
        cnt_flops = 0
        conv0 = self.conv0(x)
        if self.imagenet:
            cnt_flops += conv_flops(x, conv0, 7)
        else:
            cnt_flops += conv_flops(x, conv0, 3)
        x = conv0
        if self.imagenet:
            x = self.maxpool(x)
        for i, blocks in enumerate(self.block_list):
            for j, block in enumerate(blocks):
                conv1 = block(x)
                cnt_flops += block.cnt_flops(x)
                x = conv1
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        cnt_flops += fc_flops(x, y)
        return cnt_flops

    def forward(self, x):
        x = self.conv0(x)
        if self.imagenet:
            x = self.maxpool(x)
        for blocks in self.block_list:
            for block in blocks:
                x = block(x)
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNetChannelList(n_layer):
    if n_layer == 18:
        return [64, [[64, 64, 64], [64, 64]],
                [[128, 128, 128], [128, 128]],
                [[256, 256, 256], [256, 256]],
                [[512, 512, 512], [512, 512]]]
    elif n_layer == 20:
        return [16, [[16, 16, 16], [16, 16], [16, 16]],
                [[32, 32, 32], [32, 32], [32, 32]],
                [[64, 64, 64], [64, 64], [64, 64]]]
    elif n_layer == 32:
        return [16, [[16, 16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
                [[32, 32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
                [[64, 64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]]
    elif n_layer == 50:
        return [64, [[64, 64, 256, 256], [64, 64, 256], [64, 64, 256]],
                [[128, 128, 512, 512], [128, 128, 512], [128, 128, 512], [128, 128, 512]],
                [[256, 256, 1024, 1024], [256, 256, 1024], [256, 256, 1024], [256, 256, 1024],
                 [256, 256, 1024], [256, 256, 1024]],
                [[512, 512, 2048, 2048], [512, 512, 2048], [512, 512, 2048]]]
    elif n_layer == 56:
        return [16, [[16, 16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
                [[32, 32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
                [[64, 64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]]
    else:
        assert n_layer in cfg.keys(), 'never meet resnet_{0}'.format(n_layer)


def ResNetLite(n_layer, n_class):
    channel_list = ResNetChannelList(n_layer)
    return ResNetL(n_layer, n_class, channel_list)


if __name__ == '__main__':
    # net = ResNetLite(56, 10)
    # x = torch.zeros([16, 3, 32, 32])

    net = ResNetLite(20, 100)
    x = torch.zeros([1, 3, 32, 32])

    macs, params = profile(net, inputs=(x,))
    print(macs, params)

    flops = net.cnt_flops(x)
    print(flops)
