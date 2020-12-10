# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 21:12:16 2020

@author: enix45
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from nets.resnet import _weights_init

import math
import utils.DNAS as dnas
from nets.sr_utils import MeanShift

            
class EDSRBlockGated(nn.Module):
    def __init__(self, num_planes, res_scale = 1, dcfg = None):
        super(EDSRBlockGated, self).__init__()
        assert dcfg is not None
        self.dcfg = dcfg
        self.dcfg_nonreuse = dcfg.copy()
        self.conv1 = dnas.Conv2d(num_planes, num_planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = dnas.Conv2d(num_planes, num_planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=self.dcfg)      
        self.res_scale = res_scale

    def forward(self, x, tau=1, noise=False, reuse_prob=None):
        prob = reuse_prob
        res, p0, conv1_flops = self.conv1(x, tau, noise, p_in = prob)
        prob_list = [p0]
        flops_list = [conv1_flops]
        res = self.act(res)
        res, prob, conv2_flops = self.conv2(x, tau, noise, p_in = p0)
        prob_list.append(prob)
        flops_list.append(conv2_flops)
        res = res * self.res_scale
        res += x
        return res, prob, prob_list, flops_list

class EDSRBlockLite(nn.Module):
    def __init__(self, num_planes, res_scale = 1):
        super(EDSRBlockLite, self).__init__()
        assert len(num_planes) == 3
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_planes[0], num_planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_planes[1], num_planes[2], kernel_size=3, stride=1, padding=1, bias=False)
        m_body = [self.conv1, self.act, self.conv2]
        self.body = nn.Sequential(*m_body)
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, num_chls, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=num_chls, out_channels=num_chls * 4, kernel_size=3, stride=1, padding=1, bias=bias))
                #m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(in_channels=num_chls, out_channels=num_chls * 9, kernel_size=3, stride=1, padding=1, bias=bias))
            #m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSRGated(nn.Module):
    def __init__(self, num_blocks, num_planes, num_colors = 3, scale = 1, res_scale = 0.1):
        super(EDSRGated, self).__init__()
        self.num_blocks = num_blocks
        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)   
        self.conv0 = dnas.Conv2d(num_colors, num_planes, 3, stride=1, padding=1, bias=False, dcfg=self.dcfg)
        self.dcfg.reuse_gate = self.conv0.gate
        self.blocks = list()        
        for _ in range(num_blocks):
            self.blocks.append(EDSRBlockGated(num_planes, res_scale, self.dcfg))
        m_tail = list()
        m_tail.append(Upsampler(scale, num_planes))
        m_tail.append(nn.Conv2d(in_channels=num_planes, out_channels=num_colors, kernel_size=3, stride=1, padding=1, bias=False))
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, tau=1, noise=False):
        x = self.sub_mean(x)
        #x = self.head(x)
        x, prob, flops = self.conv0(x, tau, noise)
        prob_list = [prob]
        flops_list = [flops]
        res = x
        for i in range(self.num_blocks):
            res, prob, blk_prob_list, blk_flops_list = self.blocks[i](res, tau, noise, prob)
            flops_list += blk_flops_list
            prob_list += blk_prob_list
            prob = blk_prob_list[-1]
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x, prob_list, torch.sum(torch.stack(flops_list)), flops_list

class EDSRLite(nn.Module):
    def __init__(self, num_blocks, num_planes, channel_list, num_colors=3, scale=1, res_scale=0.1):
        super(EDSRLite, self).__init__()
        #self.num_blocks = num_blocks
        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1) 
        self.conv0 = nn.Conv2d(num_colors, channel_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        blocks = list()
        for i in range(num_blocks):
            blocks.append(EDSRBlockLite(channel_list[i+1]), res_scale)
        self.blocks = nn.Sequential(*blocks)
        m_tail = list()
        m_tail.append(Upsampler(scale, num_planes))
        m_tail.append(nn.Conv2d(in_channels=num_planes, out_channels=num_colors, kernel_size=3, stride=1, padding=1, bias=False))
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv0(x)
        res = self.blocks(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_chls, res_scale=0.1):
        super(ResBlock, self).__init__()
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=num_chls, out_channels=num_chls, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_chls, out_channels=num_chls, kernel_size=3, stride=1, padding=1, bias=False)

        m_body = [self.act1, self.conv1, self.act2, self.conv2]
        self.body = nn.Sequential(*m_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, num_blocks, num_chls, num_color=3, scale=1, res_scale=0.1):
        super(EDSR, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)

        m_head = [nn.Conv2d(in_channels=num_color, out_channels=num_chls, kernel_size=3, stride=1, padding=1, bias=False)]
        m_body = list()
        for _ in range(num_blocks):
            m_body.append(ResBlock(num_chls, res_scale))
        m_tail = list()
        m_tail.append(Upsampler(scale, num_chls))
        m_tail.append(nn.Conv2d(in_channels=num_chls, out_channels=num_color, kernel_size=3, stride=1, padding=1, bias=False))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
