# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:42:46 2020

@author: enix3
"""
import torch
import torch.nn as nn

# For DIV2K
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
def conv_flops(inputs, outputs, kernel_size):
    _, c_in, h_in, w_in = inputs.size()
    _, c_out, h_out, w_out = outputs.size()
    return kernel_size*kernel_size*c_in*c_out*h_out*w_out