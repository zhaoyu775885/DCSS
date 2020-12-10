import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

TYPE_A = 1
TYPE_B = 2

def entropy(prob):
    p = prob*(1-2e-10) + 1e-10
    return -torch.dot(p, torch.log(p))

class DcpConfig():
    def __init__(self, n_param=1, split_type=TYPE_A, reuse_gate=None):
        self.n_param = n_param
        self.split_type = split_type
        self.reuse_gate = reuse_gate

    def copy(self, reuse_gate=None):
        dcfg = copy.copy(self)
        dcfg.reuse_gate = reuse_gate
        return dcfg

    def __str__(self):
        return '{0}; {1}; {2}'.format(self.n_param, self.split_type, self.reuse_gate)


def assign_gate(n_seg, reuse_gate=None, state='uniform'):
    if reuse_gate is not None:
        return reuse_gate
    return torch.zeros([n_seg], requires_grad=True) if state == 'uniform' else \
        torch.tensor([0] * (n_seg - 1) + [1000], dtype=torch.float, requires_grad=True)


class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias, dcfg=None):
        # done: share mask is not enough,
        # done: consider wrap the additional parameters.
        # todo: degenerate to common Conv2d while dcfg is None or abnormal
        # todo: actually, dcfg == None is not allowed
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        if dcfg is None:
            return
        self.dcfg = dcfg.copy()

        if dcfg.split_type not in ['fix_seg_size', 'fix_gp_number', TYPE_A, TYPE_B]:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')

        if dcfg.split_type == 'fix_seg_size' or dcfg.split_type == TYPE_B:
            in_seg_sz = dcfg.n_param
            in_n_seg = int(np.ceil(in_planes / dcfg.n_param))
            self.seg_sz = dcfg.n_param
            self.n_seg = int(np.ceil(out_planes / dcfg.n_param))
        else:
            in_n_seg = dcfg.n_param
            in_seg_sz = int(np.ceil(in_planes / dcfg.n_param))
            self.n_seg = dcfg.n_param
            self.seg_sz = int(np.ceil(out_planes / dcfg.n_param))
            assert self.out_planes >= self.n_seg

        if in_n_seg <= self.in_planes:
            self.in_plane_list = torch.Tensor(self.__calc_seg_list(self.in_planes, in_n_seg, in_seg_sz)).cuda()
        self.out_plane_list = self.__calc_seg_list(self.out_planes, self.n_seg, self.seg_sz)
        self.mask = self.__init_mask()
        self.gate = self.__init_gate(dcfg.reuse_gate)
        self.out_plane_list = torch.Tensor(self.out_plane_list).cuda()

    def __calc_seg_list(self, planes, n_seg, seg_sz):
        seg_sz_num = planes + n_seg - n_seg * seg_sz
        seg_sub_sz_num = n_seg - seg_sz_num
        seg_list = [seg_sz] * seg_sz_num + [seg_sz - 1] * seg_sub_sz_num
        seg_tail_list = [sum(seg_list[:i + 1]) for i in range(n_seg)]
        return seg_tail_list

    def __init_mask(self):
        mask = torch.zeros(self.out_planes, self.n_seg)
        for col in range(self.n_seg):
            mask[:self.out_plane_list[col], col] = 1
        return nn.Parameter(mask, requires_grad=False)

    def __init_gate(self, reuse_gate=None):
        if reuse_gate is None:
            return nn.Parameter(assign_gate(self.n_seg, reuse_gate=reuse_gate))
        return reuse_gate

    def __cnt_flops(self, p_in, p_out, out_size):
        h, w = out_size
        def average(p, vals):
            return torch.dot(p, vals)
        cin_avg = self.in_planes if p_in is None else average(p_in, self.in_plane_list)
        cout_avg = average(p_out, self.out_plane_list)
        return cin_avg*cout_avg*h*w*self.kernel_size*self.kernel_size

    def __gumbel_softmax(self, tau=1, noise=False):
        if noise:
            uniform_noise = torch.rand(self.n_seg).cuda()
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            return F.softmax((self.gate+gumbel_noise) / tau, dim=0)
        return F.softmax((self.gate)/tau, dim=0)

    def forward(self, x, tau=1, noise=False, reuse_prob=None, p_in=None):
        y = self.conv(x)
        prob = self.__gumbel_softmax(tau, noise) if reuse_prob is None else reuse_prob
        rmask = torch.sum(self.mask * prob, dim=1)
        flops = self.__cnt_flops(p_in, prob, y.shape[2:])
        # todo: original implementation
        # return y * rmask.view(1, len(rmask), 1, 1), prob, flops
        return y, rmask, prob, flops


def weighted_feature(x, rmask):
    return x * rmask.view(1, len(rmask), 1, 1)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dcfg=None):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        self.in_features = in_features
        self.out_features = out_features
        if dcfg is None:
            print('dcfg is None')
            return
        self.dcfg = dcfg.copy()

        if dcfg.split_type not in ['fix_seg_size', 'fix_gp_number', TYPE_A, TYPE_B]:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')

        if dcfg.split_type == 'fix_seg_size' or dcfg.split_type == TYPE_B:
            in_seg_sz = dcfg.n_param
            in_n_seg = int(np.ceil(in_features / dcfg.n_param))
            self.seg_sz = dcfg.n_param
            self.n_seg = int(np.ceil(out_features / dcfg.n_param))
        else:
            in_n_seg = dcfg.n_param
            in_seg_sz = int(np.ceil(in_features / dcfg.n_param))
            self.n_seg = dcfg.n_param
            self.seg_sz = int(np.ceil(out_features / dcfg.n_param))
            assert self.out_features >= self.n_seg

        if in_n_seg <= self.in_features:
            self.in_plane_list = torch.Tensor(self.__calc_seg_list(self.in_features, in_n_seg, in_seg_sz)).cuda()
        self.out_plane_list = self.__calc_seg_list(self.out_features, self.n_seg, self.seg_sz)
        self.mask = self.__init_mask()
        self.gate = self.__init_gate(dcfg.reuse_gate)

    def __calc_seg_list(self, planes, n_seg, seg_sz):
        seg_sz_num = planes + n_seg - n_seg * seg_sz
        seg_sub_sz_num = n_seg - seg_sz_num
        seg_list = [seg_sz] * seg_sz_num + [seg_sz - 1] * seg_sub_sz_num
        seg_tail_list = [sum(seg_list[:i + 1]) for i in range(n_seg)]
        return seg_tail_list

    def __init_mask(self):
        mask = torch.zeros(self.out_features, self.n_seg)
        for col in range(self.n_seg):
            mask[:self.out_plane_list[col], col] = 1
        return nn.Parameter(mask, requires_grad=False)  # todo: determine whether to register mask

    def __init_gate(self, reuse_gate=None):
        if reuse_gate is None:
            return nn.Parameter(assign_gate(self.n_seg, reuse_gate=reuse_gate))
        return reuse_gate

    def __cnt_flops(self, p_in):
        cin_avg = self.in_features if p_in is None else torch.dot(p_in, self.in_plane_list)
        return cin_avg*self.out_features

    def forward(self, x, p_in=None):
        y = self.linear(x)
        flops = self.__cnt_flops(p_in)
        return y, flops


if __name__ == '__main__':
    # gate = torch.Tensor([0, 0, 0, 1000])
    # dcfg = DcpConfig(n_param=4, split_type=TYPE_A, reuse_gate=gate)
    # conv = Conv2d(3, 4, kernel_size=3, stride=1, padding=0, bias=False, dcfg=dcfg)
    # print(conv.mask)
    # print(conv.gate)
    #
    # x = torch.ones([1, 3, 4, 4])
    # y, gate, flops = conv(x)
    # print(y)
    # print(gate)
    # print(flops)
    #
    # for k, v in conv.named_parameters():
    #     print(k)

    logits = torch.ones(4)
    print(logits)
    loss_fn = nn.Softmax()
    prob = loss_fn(logits)
    print(prob)
    print(entropy(prob))

    # todo: to be deleted
    # you should be ashamed for this awkward loop-based implementation
    # def __cnt_flops(self, in_size, p_in, out_size, p_out):
    #     c_in, h, w = in_size
    #     assert c_in == self.in_planes
    #     if p_in is None:
    #         return self.__cnt_flops_mean(in_size, out_size, p_out)
    #     flops_list = []
    #     for i_chn in self.in_plane_list:
    #         flops_fixed_in = self.__cnt_flops_mean([i_chn, h, w], out_size, p_out)
    #         flops_list.append(flops_fixed_in)
    #     return torch.dot(torch.cuda.FloatTensor(flops_list), p_in)
    #
    # def __cnt_flops_mean(self, in_size, out_size, p_out):
    #     c_out, h_out, w_out = out_size
    #     assert c_out == self.out_planes
    #     flops_list = [self.__cnt_flops_common(in_size, torch.cuda.FloatTensor([o_chn, h_out, w_out]))
    #               for o_chn in self.out_plane_list]
    #     return torch.dot(torch.cuda.FloatTensor(flops_list), p_out)
    #
    # def __cnt_flops_common(self, in_size, out_size):
    #     c_in, h_in, w_in = in_size
    #     c_out, h_out, w_out = out_size
    #     return self.kernel_size * self.kernel_size * c_in * c_out * h_out * w_out

    # def __cnt_flops(self, p_in):
    #     if p_in is None:
    #         return self.__cnt_flops_(self.in_features, self.out_features)
    #     flops_list = [self.__cnt_flops_common(i_chn, self.out_features) for i_chn in self.in_plane_list]
    #     return torch.dot(torch.Tensor(flops_list).cuda(), p_in)
    #
    # def __cnt_flops_common(self, in_size, out_size):
    #     return in_size * out_size