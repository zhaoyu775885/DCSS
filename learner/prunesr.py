import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from timeit import default_timer as timer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import utils.DNAS as DNAS
from learner.abstract_learner import AbstractLearner
from learner.fullsr import FullSRLearner
from learner.prune import display_info
from nets.EDSR_gated import EDSRChannelList, EDSRLite


class DcpsSRLearner(AbstractLearner):
    def __init__(self, dataset, net, device, args):
        super(DcpsSRLearner, self).__init__(dataset, net, device, args)

        self.batch_size_train = self.args.batch_size
        self.batch_size_test = self.args.batch_size_test
        self.train_loader = self._build_dataloader(self.batch_size_train, is_train=True, search=True)
        self.test_loader = self._build_dataloader(self.batch_size_test, is_train=False, search=True)

        self.init_lr = self.batch_size_train / self.args.std_batch_size * self.args.std_init_lr
        print(self.init_lr)

        # setup optimizer
        self.opt_warmup = self._setup_optimizer_warmup()
        self.lr_scheduler_warmup = self._setup_lr_scheduler_warmup()

        self.opt_train = self._setup_optimizer_train()
        self.lr_scheduler_train = self._setup_lr_scheduler_train()

        self.opt_search = self._setup_optimizer_search()
        self.lr_scheduler_search = self._setup_lr_scheduler_search()

    def _setup_loss_fn(self):
        return nn.L1Loss()

    def _setup_optimizer(self):
        pass

    def _setup_optimizer_warmup(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.Adam(vars, lr=self.init_lr)

    def _setup_optimizer_train(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.Adam(vars, lr=self.init_lr*0.25)

    def _setup_optimizer_search(self):
        gates = [item[1] for item in self.forward.named_parameters() if 'gate' in item[0]]
        return optim.Adam(gates, lr=self.init_lr*0.25)

    def _setup_lr_scheduler_warmup(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[10, 20, 30], gamma=0.5)

    def _setup_lr_scheduler_train(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[10, 20, 30], gamma=0.5)

    def _setup_lr_scheduler_search(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[10, 20, 30], gamma=0.5)

    def metrics(self, pred, gt, flops=None):
        loss = self.loss_fn(pred, gt)

        loss_with_flops = loss + self.args.weight_flops * torch.log(flops)
        return loss, loss_with_flops

    def train(self, n_epoch=250, save_path='./models/slim'):
        # self.train_warmup(n_epoch=self.args.num_epoch_warmup, save_path=self.args.warmup_dir)
        # tau = self.train_search(n_epoch=self.args.num_epoch_search,
        #                         load_path=self.args.warmup_dir,
        #                         save_path=self.args.search_dir)
        tau = 0.1
        self.train_prune(tau=tau, n_epoch=n_epoch,
                         load_path=self.args.search_dir,
                         save_path=save_path)

    def train_warmup(self, n_epoch=200, save_path='./models/warmup'):
        print('Warmup', n_epoch, 'epochs')
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})

            for i, data in enumerate(self.train_loader):
                lr, hr = data[0].to(self.device), data[1].to(self.device)

                predict, prob_list, flops, flops_list = self.forward(lr, tau=1.0, noise=False)
                loss, loss_with_flops = self.metrics(predict, hr, flops)
                self.recoder.add_info(hr.size(0), {'loss': loss})
                self.opt_warmup.zero_grad()
                loss.backward()
                self.opt_warmup.step()

                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step)
                    print(i + 1,
                          ': lr={0:.1e} | loss={1:5.3f} | loss_f={2:5.3f} | flops={3} | speed={4} pic/s'.format(
                              self.opt_warmup.param_groups[0]['lr'], loss, loss_with_flops, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_warmup.step()

            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_path, 'model_'+str(epoch+1)+'.pth'))
                self.test()
                self.net.train()
        print('Finished Training')

    def train_search(self, n_epoch=100, load_path='./models/warmup', save_path='./models/search'):
        print('Search', n_epoch, 'epochs')
        self.load_model(load_path)
        self.test(tau=1.0)
        tau = 10
        total_iter = n_epoch * len(self.train_loader)
        current_iter = 0

        for epoch in range(n_epoch):
            time_prev = timer()
            print('epoch: ', epoch + 1, 'tau: ', tau)
            self.recoder.init({'loss': 0, 'loss_f': 0, 'lr': self.opt_train.param_groups[0]['lr'], 'tau':tau})

            for i, data in enumerate(self.train_loader):
                tau = 10 ** (1 - 2.0 * current_iter / (total_iter - 1))
                current_iter += 1

                # optimizing weights with training data
                lr, hr = data[0].to(self.device), data[1].to(self.device)
                self.net.train()
                predict, prob_list, flops, flops_list = self.forward(lr, tau=1.0, noise=True)
                loss, loss_with_flops = self.metrics(predict, hr, flops)
                self.opt_train.zero_grad()
                loss.backward()
                self.opt_train.step()

                # optimizing gates with searching data
                lr, hr = data[2].to(self.device), data[3].to(self.device)
                self.net.eval()
                predict, prob_list, flops, flops_list = self.forward(lr, tau=tau, noise=False)
                loss, loss_with_flops = self.metrics(predict, hr, flops)
                self.opt_search.zero_grad()
                loss_with_flops.backward()
                self.opt_search.step()

                self.recoder.add_info(hr.size(0), {'loss': loss, 'loss_f': loss_with_flops})

                if (i + 1) % 100 == 0:
                    # self.net.eval()
                    # lr, hr = data[2].to(self.device), data[3].to(self.device)
                    # self.net.eval()
                    # predict, prob_list, flops, flops_list = self.forward(lr, tau=tau, noise=False)
                    # loss, loss_with_flops = self.metrics(predict, hr)
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step)
                    print(i + 1,
                          ': lr={0:.1e} | loss={1:5.3f} | loss_f={2:5.3f} | flops={3} | speed={4} pic/s'.format(
                              self.opt_train.param_groups[0]['lr'], loss, loss_with_flops, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_train.step()
            self.lr_scheduler_search.step()

            if (epoch + 1) % 5 == 0:
                self.test(tau=tau)
                self.save_model(os.path.join(save_path, 'model_' + str(epoch + 1) + '.pth'))

        print('Finished Training')
        return tau

    def train_prune(self, tau, n_epoch=250,
                    load_path='./models/search/model.pth',
                    save_path='./models/prune/model.pth'):
        # Done, 0. load the searched model and extract the prune info
        # Done, 1. define the slim network based on prune info
        # Done, 2. train and validate, and exploit the full learner
        print('Train', n_epoch, 'epochs')
        self.load_model(load_path)
        dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
        channel_list = EDSRChannelList()

        self.net.eval()
        data = next(iter(self.train_loader))
        lr, hr = data[0].to(self.device), data[1].to(self.device)
        sr, prob_list, flops, flops_list = self.forward(lr, tau=tau, noise=False)
        display_info(flops_list, prob_list)

        # chn_list_prune = get_prune_list(channel_list, prob_list, dcfg=dcfg)
        chn_list_prune = get_uniform_prune_list(channel_list, 0.8)
        print(chn_list_prune)
        exit(1)

        net = EDSRLite(16, chn_list_prune, num_colors=3, scale=2, res_scale=0.1)

        full_learner = FullSRLearner(self.dataset, net, device=self.device, args=self.args)
        print('FLOPs:', full_learner.cnt_flops())
        full_learner.train(n_epoch=n_epoch, save_path=save_path)
        # todo: save the lite model
        # export all necessary info for slim resnet

    def test(self, tau=1.0):
        self.net.eval()
        psnrs, ssims = [], []
        flops_list, prob_list = [], []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                lr, hr = data[0].to(self.device), data[1].to(self.device)
                sr, prob_list, flops, flops_list = self.forward(lr, tau=tau, noise=False)
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
        display_info(flops_list, prob_list)
        torch.cuda.empty_cache()


def get_prune_list(channel_list_full, prob_list, dcfg, expand_rate=0):
    import numpy as np
    prune_list = []
    idx = 0

    chn_input_full, chn_output_full = 3, channel_list_full[0]
    dnas_conv = lambda input, output: DNAS.Conv2d(input, output, 1, 1, 1, False, dcfg=dcfg)
    conv = dnas_conv(chn_input_full, chn_output_full)
    chn_output_prune = int(np.round(
        min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
    ))
    chn_output_prune += int(np.ceil(expand_rate * (chn_output_full - chn_output_prune)))
    prune_list.append(chn_output_prune)
    chn_input_full = chn_output_full
    idx += 1
    for block in channel_list_full[1:-1]:
        block_list = []
        for chn_output_full in block:
            # block_prune_list = []
            conv = dnas_conv(chn_input_full, chn_output_full)
            # print(prob_list[idx], conv.out_plane_list, torch.dot(prob_list[idx], conv.out_plane_list).item())
            chn_output_prune = int(np.round(
                min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
            ))
            chn_output_prune += int(np.ceil(expand_rate * (chn_output_full - chn_output_prune)))
            # block_prune_list.append(chn_output_prune)
            block_list.append(chn_output_prune)
            chn_input_full = chn_output_full
            idx += 1

        prune_list.append(block_list)
    return prune_list


def get_uniform_prune_list(channel_list_full, prune_ratio):
    prune_list = [np.round(int(channel_list_full[0]*prune_ratio))]
    for block in channel_list_full[1:-1]:
        block_list = []
        for chn_output_full in block:
            block_list.append(np.round(int(chn_output_full*prune_ratio)))
        prune_list.append(block_list)
    return prune_list
