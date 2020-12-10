# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:09:33 2020

@author: enix45
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils.DNAS as DNAS
from timeit import default_timer as timer
from learner.abstract_learner import AbstractLearner
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

BATCH_SIZE = 32
INIT_LR = 1e-3

class DcpsLearner(AbstractLearner):
    def __init__(self, dataset, net, device='cuda:1', log_path='./log'):
        super(DcpsLearner, self).__init__(dataset, net, device, log_path)

        # dataset must be a tuple of train and test sets        
        self.train_loader = self._build_dataloader(dataset[0], BATCH_SIZE)
        self.test_loader = self._build_dataloader(dataset[1], 1)

        # setup optimizer
        self.opt_warmup = self._setup_optimizer_warmup()
        self.lr_scheduler_warmup = self._setup_lr_scheduler_warmup()

        self.opt_train = self._setup_optimizer_train()
        self.lr_scheduler_train = self._setup_lr_scheduler_train()

        self.opt_search = self._setup_optimizer_search()
        self.lr_scheduler_search = self._setup_lr_scheduler_search()

    def _build_dataloader(self, dataset, batch_size):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory=True)
        return data_loader
        
    def _loss_fn(self):
        return nn.L1Loss()

    # Weight to be modified
    def metrics(self, pred, gt, flops = None):
        loss = torch.mean(torch.abs(pred - gt))
        loss_with_flops = loss + 3 * torch.log(flops)
        return loss, loss_with_flops
    
    def _setup_optimizer_warmup(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        #return optim.SGD(vars, lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)
        return optim.Adam(vars, lr = INIT_LR)

    def _setup_optimizer_train(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        #return optim.SGD(vars, lr=INIT_LR * 0.1, momentum=MOMENTUM, weight_decay=L2_REG)
        return optim.Adam(vars, lr = INIT_LR * 0.1)

    def _setup_optimizer_search(self):
        # todo: the risk that ignores the shared gates in the network
        gates = [item[1] for item in self.forward.named_parameters() if 'gate' in item[0]]
        return optim.Adam(gates, lr=0.01)

    def _setup_lr_scheduler_warmup(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[100, 150], gamma=0.1)

    def _setup_lr_scheduler_train(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[50, 100], gamma=0.1)

    def _setup_lr_scheduler_search(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[50, 100], gamma=0.1)
    
    def train_warmup(self, num_epoch, save_path):
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})
            for ii, data in enumerate(data_loader):
                in_data, im_gt = [x.to(self.device) for x in data]
                outputs, prob_list, flops, flops_list = self.forward(in_data)
                loss, loss_with_flops = self.metrics(outputs, im_gt, flops)
                loss.requires_grad_()
                loss_with_flops.requires_grad_()
                self.recoder.add_info(im_gt.size(0), {'loss': loss})
                self.opt_warmup.zero_grad()
                loss.backward()
                self.opt_warmup.step()
                if (ii + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * BATCH_SIZE / time_step)
                    print(i + 1, ': lr={0:.1e} | loss={2:5.3f} | flops={3} | speed={4} pic/s'.format(
                        self.opt_warmup.param_groups[0]['lr'], loss, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_warmup.step()
            if (epoch + 1) % 10 == 0:
                self.save_model(path=save_path)
                #self.test_dcps()
                self.net.train()
        print('Finished Warming-up')
        
    def train_search(self, num_epoch, load_path, save_path):
        self.load_model(path = load_path)
        tau = 1.0
        for epoch in range(num_epoch):
            tau = 10 ** (1 - 2. * epoch / (num_epoch - 1))
            print('epoch: ', epoch + 1, 'tau: ', tau)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'loss_f': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})
            for ii, data in enumerate(self.train_loader):
                in_data, im_gt = [x.to(self.device) for x in data]
                if ii % 2 == 1:
                    self.net.eval()
                    outputs, prob_list, flops, flops_list = self.forward(in_data, tau = tau, noise = False)
                    loss, loss_with_flops = self.metrics(outputs, im_gt, flops)
                    self.opt_search.zero_grad()
                    loss_with_flops.backward()
                    self.opt_search.step()
                else:
                    self.net.train()
                    outputs, prob_list, flops, flops_list = self.forward(in_data, tau = tau, noise = True)
                    loss, loss_with_flops = self.metrics(outputs, im_gt, flops)
                    self.opt_train.zero_grad()
                    loss.backward()
                    self.opt_train.step()
                
                self.recoder.add_info(im_gt.size(0), {'loss': loss, 'loss_f': loss_with_flops})
                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * BATCH_SIZE / time_step)
                    print(i + 1,
                          ': lr={0:.1e} |'.format(self.opt_train.param_groups[0]['lr']),
                          'loss={0:5.3f} | loss_f={1:5.3f} | flops={2} | speed={3} pic/s'.format(
                              loss, loss_with_flops, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_train.step()
            self.lr_scheduler_search.step()
            if (epoch + 1) % 10 == 0:
                self.save_model(path=save_path)
                #self.test_dcps(tau=tau)
                display_info(flops_list, prob_list)
        print('Finished Training')
        return tau
        
    def train_prune(self, tau, n_epoch=250, load_path, save_path):
        self.load_model(load_path)
        dcfg = DNAS.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
        data = next(iter(self.train_loader))
        in_data, im_gt = [x.to(self.device) for x in data]
        self.net.eval()
        outputs, prob_list, flops, flops_list = self.forward(in_data, tau=tau, noise=False)
        
    def test(self, tau):
        self.net.eval()
        torch.set_grad_enabled(False)
        psnrs = list()
        ssims = list()
        for ii, data in enumerate(self.test_loader):
            lr, hr = [x.to(self.device) for x in data]
            sr = self.net(lr)
            sr = torch.clamp(sr, 0, 1)
            sr = sr.cpu().detach().numpy() * 255
            hr = hr.cpu().detach().numpy() * 255
            sr = np.transpose(sr.squeeze(), (1, 2, 0))
            hr = np.transpose(hr.squeeze(), (1, 2, 0))
            sr = sr.astype(np.uint8)
            hr = hr.astype(np.uint8)
            psnr = compare_psnr(hr, sr, data_range = 255)
            ssim = compare_ssim(hr, sr, data_range = 255, multichannel = True)
            psnrs.append(psnr)
            ssims.append(ssim)
        print('PSNR= {2:.4f}, SSIM= {2:.4f}'.format(np.mean(psnrs), np.mean(ssims)))
        self.net.train()
        torch.set_grad_enabled(True)
        
def display_info(flops_list, prob_list):
    print('=============')
    for flops, prob in zip(flops_list, prob_list):
        pinfo = ''
        for item in prob.tolist():
            pinfo += '{0:.3f} '.format(item)
        pinfo += ', {0:.0f}'.format(flops)
        print(pinfo)
    print('-------------\n')
