# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:12:16 2020

@author: enix45
"""

import torch
import torch.nn as nn

import math
import utils.DNAS as dnas
from nets.sr_utils import MeanShift
from nets.resnet_lite import conv_flops
from thop import profile


class Upsampler(nn.Sequential):
    def __init__(self, scale, num_chls, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=num_chls, out_channels=num_chls * 4, kernel_size=3, stride=1, padding=1,
                                   bias=bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(in_channels=num_chls, out_channels=num_chls * 9, kernel_size=3, stride=1, padding=1,
                               bias=bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSRBlockGated(nn.Module):
    def __init__(self, in_plane, out_planes, dcfg, res_scale: float = 1):
        super(EDSRBlockGated, self).__init__()
        assert len(out_planes) == 2
        assert out_planes[-1] == in_plane
        assert dcfg is not None
        # self.dcfg = dcfg
        # self.dcfg_nonreuse = dcfg.copy()

        # self.act1 = nn.ReLU(inplace=True)
        self.conv1 = dnas.Conv2d(in_plane, out_planes[0], kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=dcfg.copy())
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = dnas.Conv2d(out_planes[0], in_plane, kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=dcfg)
        self.res_scale = res_scale

    def forward(self, x, tau=1, noise=False, reuse_prob=None):
        prob = reuse_prob

        res = x
        # res = self.act1(x)
        res, rmask_1, p1, conv1_flops = self.conv1(res, tau, noise, p_in=prob)
        res = dnas.weighted_feature(res, rmask_1)
        prob_list = [p1]
        flops_list = [conv1_flops]

        res = self.act2(res)
        res, rmask_2, p2, conv2_flops = self.conv2(res, tau, noise, reuse_prob=prob, p_in=p1)
        res = dnas.weighted_feature(res, rmask_2)
        prob_list.append(prob)
        flops_list.append(conv2_flops)

        res = res * self.res_scale + x
        return res, rmask_2, prob, prob_list, flops_list


class EDSRGated(nn.Module):
    def __init__(self, num_blocks, channel_list, dcfg, num_colors=3, scale=1, res_scale=0.1):
        super(EDSRGated, self).__init__()
        self.num_blocks = num_blocks
        self.dcfg = dcfg

        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)

        self.conv0 = dnas.Conv2d(num_colors, channel_list[0], 3, stride=1, padding=1, bias=False, dcfg=self.dcfg)
        self.dcfg.reuse_gate = self.conv0.gate

        blocks = list()
        in_plane = channel_list[0]
        for i in range(num_blocks):
            blocks.append(EDSRBlockGated(in_plane, channel_list[i + 1], self.dcfg, res_scale))
            in_plane = channel_list[i + 1][-1]
        self.blocks = nn.Sequential(*blocks)

        self.tail = Upsampler(scale, in_plane)
        self.output = nn.Conv2d(in_channels=in_plane, out_channels=num_colors, kernel_size=3, stride=1, padding=1,
                                bias=False)

    def forward(self, x, tau=1, noise=False):
        x = self.sub_mean(x)
        x, rmask, prob, flops = self.conv0(x, tau, noise)
        x = dnas.weighted_feature(x, rmask)
        prob_list, flops_list = [prob], [flops]
        res = x
        for i in range(self.num_blocks):
            res, rmask, prob, blk_prob_list, blk_flops_list = self.blocks[i](res, tau, noise, prob)
            flops_list += blk_flops_list
            prob_list += blk_prob_list
            # todo: seems redundant
            prob = blk_prob_list[-1]
        res += x
        x = dnas.weighted_feature(self.tail(res), rmask)
        x = self.output(x)
        x = self.add_mean(x)
        return x, prob_list, torch.sum(torch.stack(flops_list)), flops_list


class EDSRBlockLite(nn.Module):
    def __init__(self, in_plane, out_planes, res_scale: float = 1):
        super(EDSRBlockLite, self).__init__()
        assert len(out_planes) == 2
        assert out_planes[-1] == in_plane
        # self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_plane, out_planes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes[0], in_plane, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_scale = res_scale

    def cnt_flops(self, x):
        flops = 0
        res = x
        conv1 = self.conv1(res)
        flops += conv_flops(res, conv1, 3)
        conv2 = self.conv2(self.act2(conv1))
        flops += conv_flops(conv1, conv2, 3)
        return flops

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(self.act2(res)).mul(self.res_scale)
        res += x
        return res


class EDSRLite(nn.Module):
    def __init__(self, num_blocks, channel_list, num_colors=3, scale=1, res_scale=0.1):
        super(EDSRLite, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)

        self.conv0 = nn.Conv2d(num_colors, channel_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        blocks = list()
        in_plane = channel_list[0]
        for i in range(num_blocks):
            blocks.append(EDSRBlockLite(in_plane, channel_list[i + 1], res_scale))
            in_plane = channel_list[i + 1][-1]
        self.blocks = nn.ModuleList(blocks)

        self.upsampler = Upsampler(scale, in_plane)
        self.tail = nn.Conv2d(in_channels=in_plane, out_channels=num_colors, kernel_size=3, stride=1,
                              padding=1, bias=False)

    def cnt_flops(self, x):
        flops = 0
        x = self.sub_mean(x)
        conv0 = self.conv0(x)
        flops += conv_flops(x, conv0, 3)
        x = conv0
        res = conv0
        for block in self.blocks:
            flops += block.cnt_flops(res)
            res = block(res)
        res += x
        x = self.upsampler(res)
        y = self.tail(x)
        return flops

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv0(x)
        res = x
        for block in self.blocks:
            res = block(res)
        res += x
        x = self.upsampler(res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_chls, res_scale=0.1):
        super(ResBlock, self).__init__()
        # self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=num_chls, out_channels=num_chls, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_chls, out_channels=num_chls, kernel_size=3, stride=1, padding=1,
                               bias=False)

        # m_body = [self.act1, self.conv1, self.act2, self.conv2]
        m_body = [self.conv1, self.act2, self.conv2]
        self.body = nn.Sequential(*m_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, num_blocks, num_chls, num_colors=3, scale=1, res_scale=0.1):
        super(EDSR, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)

        self.conv0 = nn.Conv2d(in_channels=num_colors, out_channels=num_chls, kernel_size=3, stride=1, padding=1,
                               bias=False)
        m_body = list()
        for _ in range(num_blocks):
            m_body.append(ResBlock(num_chls, res_scale))
        m_tail = list()
        m_tail.append(Upsampler(scale, num_chls))
        m_tail.append(
            nn.Conv2d(in_channels=num_chls, out_channels=num_colors, kernel_size=3, stride=1, padding=1, bias=False))

        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv0(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


def EDSRChannelList():
    channel_list = [64,
                    [64, 64], [64, 64], [64, 64], [64, 64],
                    [64, 64], [64, 64], [64, 64], [64, 64],
                    [64, 64], [64, 64], [64, 64], [64, 64],
                    [64, 64], [64, 64], [64, 64], [64, 64],
                    64]
    return channel_list


def EDSRDcps(num_blocks, num_colors=3, scale=1, res_scale=0.1):
    dcfg = dnas.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
    chn_list = EDSRChannelList()
    return EDSRGated(num_blocks=num_blocks, channel_list=chn_list, dcfg=dcfg, num_colors=num_colors,
                     scale=scale, res_scale=res_scale)


if __name__ == '__main__':
    net = EDSR(num_blocks=16, num_chls=64, num_colors=3, scale=2, res_scale=0.1)
    x = torch.zeros([1, 3, 48, 48])
    # y = net(x)
    # print(net)
    # exit(1)
    macs, params = profile(net, inputs=(x,))
    print(macs, params)
    # print(x.shape)
    # print(y.shape)

    chn_list = EDSRChannelList()
    net_2 = EDSRLite(16, chn_list, num_colors=3, scale=2, res_scale=0.1)
    g = net_2(x)
    macs, params = profile(net_2, inputs=(x,))
    flops = net_2.cnt_flops(x)
    print(flops)
    # print(macs, params)
    # print(g.shape)
