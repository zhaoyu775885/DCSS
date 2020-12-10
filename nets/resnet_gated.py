import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.DNAS as DNAS
from nets.resnet import _weights_init
from nets.resnet_lite import ResNetChannelList

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

_BATCH_NORM_DECAY = 0.01
_EPSILON = 1e-5

'''

This implementation is to weight the convolution after the BN layer.

'''


class ResidualBlockGated(nn.Module):
    """
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    """

    def __init__(self, in_planes, out_planes_list, stride=2, project=False, dcfg=None):
        super(ResidualBlockGated, self).__init__()

        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        assert dcfg is not None
        self.dcfg = dcfg
        self.dcfg_nonreuse = dcfg.copy()
        self.bn0 = nn.BatchNorm2d(in_planes, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv1 = DNAS.Conv2d(in_planes, out_planes_1, kernel_size=3, stride=stride, padding=1, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.bn1 = nn.BatchNorm2d(out_planes_1, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.shortcut = None
        # if stride != 1 or len(out_planes_list)>2 or in_planes != out_planes_2:
        if project:
            self.shortcut = DNAS.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride, padding=0,
                                        bias=False, dcfg=self.dcfg_nonreuse)
            self.dcfg.reuse_gate = self.shortcut.gate
        self.conv2 = DNAS.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=self.dcfg)

    def forward(self, x, tau=1, noise=False, reuse_prob=None, rmask=None):
        prob = reuse_prob
        shortcut = x
        # x = self.bn0(x), x = DNAS.weighted_feature(x, rmask), x = F.relu(x)
        x = F.relu(DNAS.weighted_feature(self.bn0(x), rmask))
        prob_list, flops_list = [], []
        if self.shortcut is not None:
            # todo: original implementation
            # shortcut, prob, shortcut_flops = self.shortcut(x, tau, noise, p_in=prob)
            shortcut, rmask, prob, shortcut_flops = self.shortcut(x, tau, noise, p_in=prob)
            prob_list.append(prob)
            flops_list.append(shortcut_flops)
        # todo: original implementation
        # x, p0, conv1_flops = self.conv1(x, tau, noise, p_in=prob)
        x, rmask_1, p1, conv1_flops = self.conv1(x, tau, noise, p_in=prob)
        # x = self.bn1(x), x = DNAS.weighted_feature(x, rmask_1), x = F.relu(x)
        x = F.relu(DNAS.weighted_feature(self.bn1(x), rmask_1))
        prob_list.insert(0, p1)
        flops_list.insert(0, conv1_flops)

        # todo: original implementation
        # x, prob, conv2_flops = self.conv2(x, tau, noise, reuse_prob=prob, p_in=p0)
        x, rmask_2, prob, conv2_flops = self.conv2(x, tau, noise, reuse_prob=prob, p_in=p1)
        prob_list.insert(1, prob)
        flops_list.insert(1, conv2_flops)

        x += shortcut
        x = DNAS.weighted_feature(x, rmask)
        # todo: the order of prob and flops should correspond to the order of channels
        # todo: original implementation
        # return x, prob, prob_list, flops_list
        return x, rmask_2, prob, prob_list, flops_list


class BottleneckGated(nn.Module):
    def __init__(self, in_planes, out_planes_list, stride=2, project=False, dcfg=None):
        super(BottleneckGated, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        out_planes_3 = out_planes_list[2]
        assert dcfg is not None
        self.dcfg = dcfg
        self.dcfg_nonreuse = dcfg.copy()

        self.bn0 = nn.BatchNorm2d(in_planes, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv1 = DNAS.Conv2d(in_planes, out_planes_1, kernel_size=1, stride=1, padding=0, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.bn1 = nn.BatchNorm2d(out_planes_1, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.conv2 = DNAS.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=stride, padding=1, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.bn2 = nn.BatchNorm2d(out_planes_2, momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.shortcut = None
        # if stride != 1 and len(out_planes_list) > 2:
        if project:
            self.shortcut = DNAS.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride,
                                        padding=0, bias=False, dcfg=self.dcfg_nonreuse)
            self.dcfg.reuse_gate = self.shortcut.gate
        self.conv3 = DNAS.Conv2d(out_planes_2, out_planes_3, kernel_size=1, stride=1, padding=0, bias=False,
                                 dcfg=self.dcfg)

    def forward(self, x, tau=1, noise=False, reuse_prob=None, rmask=None):
        prob = reuse_prob
        shortcut = x
        x = F.relu(DNAS.weighted_feature(self.bn0(x), rmask))
        prob_list, flops_list = [], []
        if self.shortcut is not None:
            shortcut, rmask, prob, shortcut_flops = self.shortcut(x, tau, noise, p_in=prob)
            prob_list.append(prob)
            flops_list.append(shortcut_flops)

        x, rmask_1, p1, conv1_flops = self.conv1(x, tau, noise, p_in=prob)
        x = F.relu(DNAS.weighted_feature(self.bn1(x), rmask_1))
        prob_list.insert(0, p1)
        flops_list.insert(0, conv1_flops)

        x, rmask_2, p2, conv2_flops = self.conv2(x, tau, noise, p_in=p1)
        x = F.relu(DNAS.weighted_feature(self.bn2(x), rmask_2))
        prob_list.insert(1, p2)
        flops_list.insert(1, conv2_flops)

        x, rmask_3, prob, conv3_flops = self.conv3(x, tau, noise, reuse_prob=prob, p_in=p2)
        prob_list.insert(2, prob)
        flops_list.insert(2, conv3_flops)

        x += shortcut
        x = DNAS.weighted_feature(x, rmask)

        return x, rmask_3, prob, prob_list, flops_list


class ResNet(nn.Module):
    def __init__(self, n_layer, n_class, channel_lists, dcfg):
        super(ResNet, self).__init__()

        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.n_class = n_class
        self.channel_lists = channel_lists
        self.block_n_cell = cfg[n_layer]
        self.dcfg = dcfg
        self.base_n_channel = channel_lists[0]
        self.imagenet = len(self.block_n_cell) > 3

        if self.imagenet:
            self.cell_fn = BottleneckGated if n_layer >= 50 else ResidualBlockGated
            self.conv0 = DNAS.Conv2d(3, self.base_n_channel, 7, stride=2, padding=3, bias=False, dcfg=self.dcfg)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.cell_fn = ResidualBlockGated
            self.conv0 = DNAS.Conv2d(3, self.base_n_channel, 3, stride=1, padding=1, bias=False, dcfg=self.dcfg)
        self.dcfg.reuse_gate = self.conv0.gate

        self.block_list = self._block_layers()
        self.bn = nn.BatchNorm2d(channel_lists[-1][-1][-1], momentum=_BATCH_NORM_DECAY, eps=_EPSILON)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DNAS.Linear(channel_lists[-1][-1][-1], self.n_class, dcfg=self.dcfg)
        self.apply(_weights_init)

    def _block_fn(self, in_planes, out_planes_lists, n_cell, strides):
        blocks = [self.cell_fn(in_planes, out_planes_lists[0], strides, project=True, dcfg=self.dcfg)]
        for i in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes_lists[i - 1][-1], out_planes_lists[i], 1, dcfg=self.dcfg))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i == 0:
                block = self._block_fn(self.base_n_channel, self.channel_lists[i + 1], n_cell, 1)
                block_list.append(block)
            else:
                in_planes = self.channel_lists[i][-1][-1]
                block = self._block_fn(in_planes, self.channel_lists[i + 1], n_cell, 2)
                block_list.append(block)
        return nn.ModuleList(block_list)

    def forward(self, x, tau=1, noise=False):
        x, rmask, prob, flops = self.conv0(x, tau, noise)
        # todo: original implementation
        # x, prob, flops = self.conv0(x, tau, noise)
        prob_list, flops_list = [prob], [flops]
        if self.imagenet:
            x = self.maxpool(x)
        for i, blocks in enumerate(self.block_list):
            for block in blocks:
                # todo: original implementation
                # x, prob, blk_prob_list, blk_flops_list = block(x, tau, noise, reuse_prob=prob, rmask=rmask)
                x, rmask, prob, blk_prob_list, blk_flops_list = block(x, tau, noise, reuse_prob=prob, rmask=rmask)
                prob_list += blk_prob_list
                flops_list += blk_flops_list
                prob = blk_prob_list[-1]
        x = F.relu(DNAS.weighted_feature(self.bn(x), rmask))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x, fc_flops = self.fc(x, p_in=prob)
        flops_list += [fc_flops]
        return x, prob_list, torch.sum(torch.stack(flops_list)), flops_list

# def ResNet20Gated(n_classes):
#     dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
#     channel_list_20 = ResNetChannelList(20)
#     return ResNet(20, n_classes, channel_list_20, dcfg)
#
#
# def ResNet32Gated(n_classes):
#     dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
#     channel_list_32 = ResNetChannelList(32)
#     return ResNet(32, n_classes, channel_list_32, dcfg)
#
#
# def ResNet56Gated(n_classes):
#     dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
#     channel_list_56 = ResNetChannelList(56)
#     return ResNet(56, n_classes, channel_list_56, dcfg)

def ResNetGated(n_layer, n_class):
    dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
    channel_list = ResNetChannelList(n_layer)
    return ResNet(n_layer, n_class, channel_list, dcfg)


if __name__ == '__main__':

    # net = ResNetLite(56, 10)
    # x = torch.zeros([16, 3, 32, 32])

    net = ResNetGated(20, 100)
    x = torch.zeros([1, 3, 32, 32])
    y = net(x)
    print(y.shape)