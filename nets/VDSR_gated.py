# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:28:04 2020

@author: enix45
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from nets.resnet import _weights_init

import dcps.DNAS as dnas
from sr_utils import MeanShift

class VDSRLayerGated(nn.Module):
    def __init__(self, in_planes = 64, out_planes = 64, dcfg = None):
        super(VDSRLayerGated, self).__init__()
        assert dcfg is not None
        self.dcfg = dcfg
        self.dcfg_nonreuse = dcfg.copy()
        self.conv1 = dnas.Conv2d(in_planes, out_planes, kernel_size=3, stride = 1, padding=1, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, tau=1, noise=False, reuse_prob=None):
        prob = reuse_prob
        x, p0, conv1_flops = self.conv1(x, tau, noise, p_in=prob)
        x = self.relu(x)
        return x, prob, p0, conv1_flops

class VDSRLayerLite(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(VDSRLayerLite, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)
                            
    def forward(self, x):
        return self.relu(self.conv1(x))

class VDSRGated(nn.Module):
    def __init__(self, num_layers = 18, num_colors = 3, num_planes = 64, dcfg = None):
        super(VDSRGated, self).__init__()
        self.dcfg = dcfg
        self.num_layers = num_layers
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)
        
        self.conv0 = dnas.Conv2d(num_colors, num_planes, kernel_size=3, stride=1, padding=1, bias=False, dcfg=self.dcfg)
        self.relu = nn.ReLU(inplace=True)
        self.dcfg.reuse_gate = self.conv0.gate
        
        self.layer_list = list()
        for _ in range(num_layers - 2):
            self.layer_list.append(VDSRLayerGated(num_planes, num_planes, self.dcfg))
        self.layer1 = nn.Conv2d(in_channels=num_planes, out_channels=num_colors, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x, tau=1, noise=False):
        x = self.sub_mean(x)
        res = x
        res, prob, flops = self.conv0(res, tau, noise)
        res = self.relu(res)
        prob_list = [prob]
        flops_list = [flops]
        for i in range(self.num_layers - 2):
            res, prob, p0, flops = self.layer_list[i](res, tau, noise, prob)
            prob_list.append(p0)
            flops_list.append(flops)
        res = self.layer1(res)
        x += res
        x = self.add_mean(x)
        return x, prob_list, torch.sum(torch.stack(flops_list)), flops_list

class VDSRLite(nn.Module):
    def __init__(self, num_layers, num_colors, channel_list):
        super(VDSRLite, self).__init__()
        self.num_layers = num_layers
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)
        self.conv0 = nn.Conv2d(num_colors, channel_list[0], kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.layer_list = list()
        for i in range(num_layers - 2):
            self.layer_list.append(VDSRLayerLite(channel_list[i], channel_list[i+1]))
        self.layer_1 = nn.Conv2d(channel_list[-1], num_colors, kernel_size = 3, stride = 1, padding = 1, bias = False)

    def forward(self, x):
        x = self.sub_mean(x)
        res = x
        res = self.relu(self.conv0(res))
        for i in range(self.num_layers - 2):
            res = self.layer_list[i](res)
        x += res
        x = self.add_mean(x)
        return x
    
