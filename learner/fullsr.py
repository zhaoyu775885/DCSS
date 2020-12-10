import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from learner.abstract_learner import AbstractLearner
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os


class FullSRLearner(AbstractLearner):
    def __init__(self, dataset, net, device, args):
        super(FullSRLearner, self).__init__(dataset, net, device, args)
        self.batch_size_train = self.args.batch_size
        self.batch_size_test = self.args.batch_size_test
        self.train_loader = self._build_dataloader(self.batch_size_train, is_train=True)
        self.test_loader = self._build_dataloader(self.batch_size_test, is_train=False)

        self.init_lr = self.batch_size_train / self.args.std_batch_size * self.args.std_init_lr
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    def _setup_loss_fn(self):
        return nn.L1Loss()

    def _setup_optimizer(self):
        return optim.Adam(self.forward.parameters(), lr=self.init_lr)

    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                    milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                                                    gamma=0.5)

    def metrics(self, predict, truth):
        # loss = torch.mean(torch.abs(predict-truth))
        loss = self.loss_fn(predict, truth)
        return loss

    def cnt_flops(self):
        flops = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            flops = self.forward.cnt_flops(inputs)
            break
        return flops

    def train(self, n_epoch, save_path='./models/full/model.pth'):
        self.net.train()
        print(n_epoch)
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'lr': self.opt.param_groups[0]['lr']})

            for i, data in enumerate(self.train_loader):
                # print(i)
                lr, hr = data[0].to(self.device), data[1].to(self.device)
                predict = self.forward(lr)
                loss = self.metrics(predict, hr)
                self.recoder.add_info(hr.size(0), {'loss': loss})
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step)
                    print(i + 1, ': lr={0:.1e} | loss={1:6.4f} | speed={2} pic/s'.format(
                        self.opt.param_groups[0]['lr'], loss, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler.step()

            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_path, 'model_'+str(epoch+1)+'.pth'))
                self.test()
                self.net.train()
        print('Finished Training')

    def test(self):
        self.net.eval()
        # torch.set_grad_enabled(False)
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
            psnr = compare_psnr(hr, sr, data_range=255)
            ssim = compare_ssim(hr, sr, data_range=255, multichannel=True)
            psnrs.append(psnr)
            ssims.append(ssim)
        print('PSNR= {0:.4f}, SSIM= {1:.4f}'.format(np.mean(psnrs), np.mean(ssims)))
        # self.net.train()
        # torch.set_grad_enabled(True)
