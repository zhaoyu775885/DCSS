import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from nets.resnet_lite import ResNetL
from learner.abstract_learner import AbstractLearner
from learner.full import FullLearner
from learner.distiller import Distiller
import utils.DNAS as DNAS
from utils.DNAS import entropy
import os
from itertools import cycle
from nets.resnet_lite import ResNetChannelList


# BATCH_SIZE = 128
# INIT_LR = 1e-1
# MOMENTUM = 0.9
# L2_REG = 5e-4


class DcpsLearner(AbstractLearner):
    def __init__(self, dataset, net, device, args, teacher=None):
        super(DcpsLearner, self).__init__(dataset, net, device, args)

        self.batch_size_train = self.args.batch_size
        self.batch_size_test = self.args.batch_size_test
        self.train_loader = self._build_dataloader(self.batch_size_train, is_train=True, search=True)
        self.test_loader = self._build_dataloader(self.batch_size_test, is_train=False, search=True)

        self.init_lr = self.batch_size_train / self.args.std_batch_size * self.args.std_init_lr
        # setup optimizer
        self.opt_warmup = self._setup_optimizer_warmup()
        self.lr_scheduler_warmup = self._setup_lr_scheduler_warmup()

        self.opt_train = self._setup_optimizer_train()
        self.lr_scheduler_train = self._setup_lr_scheduler_train()

        self.opt_search = self._setup_optimizer_search()
        self.lr_scheduler_search = self._setup_lr_scheduler_search()

        self.teacher = teacher

    def _setup_loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        pass

    def _setup_optimizer_warmup(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=self.init_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def _setup_optimizer_train(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=self.init_lr*0.1, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def _setup_optimizer_search(self):
        gates = [item[1] for item in self.forward.named_parameters() if 'gate' in item[0]]
        return optim.Adam(gates, lr=self.init_lr*0.1)

    def _setup_lr_scheduler_warmup(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[100, 150], gamma=0.1)

    def _setup_lr_scheduler_train(self):
        # return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_train, T_max=150, eta_min=1e-4)
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[50, 100], gamma=0.1)

    def _setup_lr_scheduler_search(self):
        # return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_search, T_max=150, eta_min=1e-4)
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[50, 100], gamma=0.1)

    def metrics(self, outputs, labels, flops=None, prob_list=None):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.loss_fn(outputs, labels)
        prob_loss = 0
        if prob_list is not None:
            for prob in prob_list:
                prob_loss += entropy(prob)
            loss += 0.00*prob_loss
        tolerance = 0.01
        target_flops = 20000000
        coef = 0.1
        # if flops < (1 - tolerance) * target_flops:
        #     coef = -10
        # elif flops > (1 + tolerance) * target_flops:
        #     coef = 10
        loss_with_flops = loss + coef * torch.log(flops)
        accuracy = correct / labels.size(0)
        return accuracy, loss, loss_with_flops

    def train(self, n_epoch=250, save_path='./models/slim'):
        # self.train_warmup(n_epoch=150, save_path=self.args.warmup_dir)
        # tau = self.train_search(n_epoch=150,
        #                         load_path=self.args.warmup_dir,
        #                         save_path=self.args.search_dir)
        tau = 0.1
        self.train_prune(tau=tau, n_epoch=n_epoch,
                         load_path=self.args.search_dir,
                         save_path=save_path)

    def train_warmup(self, n_epoch=200, save_path='./models/warmup'):
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'accuracy': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs, _, flops, flops_list = self.forward(inputs, tau=1.0, noise=False)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.recoder.add_info(labels.size(0), {'loss': loss, 'accuracy': accuracy})
                self.opt_warmup.zero_grad()
                loss.backward()
                self.opt_warmup.step()
                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step)
                    print(i + 1, ': lr={0:.1e} | acc={1:5.2f} | loss={2:5.2f} | flops={3} | speed={4} pic/s'.format(
                        self.opt_warmup.param_groups[0]['lr'], accuracy * 100, loss, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_warmup.step()
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_path, 'model_'+str(epoch+1)+'.pth'))
                self.test(tau=1.0)
                self.net.train()
        print('Finished Warming-up')

    def train_search(self, n_epoch=100, load_path='./models/warmup', save_path='./models/search'):
        self.load_model(load_path)
        self.test(tau=1.0)
        tau = 10
        total_iter = n_epoch * len(self.train_loader)
        current_iter = 0

        for epoch in range(n_epoch):
            time_prev = timer()
            # tau = 10 ** (1 - 2 * epoch / (n_epoch - 1))
            print('epoch: ', epoch + 1, ' tau: ', tau)
            self.recoder.init({'loss': 0, 'loss_f': 0, 'accuracy': 0,
                               'lr': self.opt_train.param_groups[0]['lr'],
                               'tau': tau})

            for i, data in enumerate(self.train_loader):
                # tau = 10 - (10-0.1) / (total_iter-1) * current_iter
                tau = 10 ** (1 - 2.0 * current_iter / (total_iter-1))
                current_iter += 1

                # optimizing weights with training data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.net.train()
                outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=True)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.opt_train.zero_grad()
                loss.backward()
                self.opt_train.step()

                # optimizing gates with searching data
                inputs, labels = data[2].to(self.device), data[3].to(self.device)
                self.net.eval()
                outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.opt_search.zero_grad()
                loss_with_flops.backward()
                self.opt_search.step()

                self.recoder.add_info(labels.size(0), {'loss': loss, 'loss_f': loss_with_flops,
                                                       'accuracy': accuracy})
                if (i + 1) % 100 == 0:
                    self.net.eval()
                    inputs, labels = data[2].to(self.device), data[3].to(self.device)
                    outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
                    accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step)
                    print(i + 1,
                          ': lr={0:.1e} | acc={1:5.2f} |'.format(self.opt_train.param_groups[0]['lr'], accuracy * 100),
                          'loss={0:5.2f} | loss_f={1:5.2f} | flops={2} | speed={3} pic/s'.format(
                              loss, loss_with_flops, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_train.step()
            self.lr_scheduler_search.step()
            if (epoch + 1) % 10 == 0:
                self.test(tau=tau)

            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_path, 'model_' + str(epoch + 1) + '.pth'))

        print('Finished Training')
        return tau

    def train_prune(self, tau, n_epoch=250,
                    load_path='./models/search/model.pth',
                    save_path='./models/prune/model.pth'):
        # Done, 0. load the searched model and extract the prune info
        # Done, 1. define the slim network based on prune info
        # Done, 2. train and validate, and exploit the full learner
        self.load_model(load_path)
        dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
        channel_list = ResNetChannelList(self.args.teacher_net_index)

        self.net.eval()
        data = next(iter(self.train_loader))
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
        print('=================')
        print(tau)
        for prob in prob_list:
            for item in prob.tolist():
                print('{0:.2f}'.format(item), end=' ')
            print()
        print('------------------')

        for item in self.forward.named_parameters():
            # if '0.0.bn0' in item[0] and 'bias' not in item[0]:
            #     print(item)
            if 'conv0.conv.weight' in item[0]:
                print(item[1][:, 0, 0, 0])
                print(item[1][:, 1, 2, 1])

        channel_list_prune = get_prune_list(channel_list, prob_list, dcfg=dcfg)
        # channel_list_prune = [16,
        #                       [[10, 12, 12], [7, 12], [11, 12]],
        #                       [[31, 25, 25], [19, 25], [32, 25]],
        #                       [[39, 43, 43], [61, 43], [41, 43]]]

        print(channel_list_prune)
        # exit(1)
        # channel_list_prune = [13, [[9, 13], [9, 13], [16, 13]], [[16, 25, 25], [27, 25], [16, 25]], [[54, 64, 64], [45, 64], [29, 64]]]
        # teacher_net = ResNet20(n_classes=self.dataset.n_class)
        # teacher = Distiller(self.dataset, teacher_net, self.device, self.args, model_path='./models/6884.pth')

        net = ResNetL(self.args.net_index, self.dataset.n_class, channel_list_prune)
        full_learner = FullLearner(self.dataset, net, device=self.device, args=self.args, teacher=self.teacher)
        print('FLOPs:', full_learner.cnt_flops())
        full_learner.train(n_epoch=n_epoch, save_path=save_path)
        # todo: save the lite model
        # export all necessary info for slim resnet

    def test(self, tau=1.0):
        self.net.eval()
        total_accuracy_sum = 0
        total_loss_sum = 0
        for i, data in enumerate(self.test_loader, 0):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs, prob_list, flops, flops_list = self.forward(images, tau=tau, noise=False)
            # todo: to be fixed
            accuracy, loss, _ = self.metrics(outputs, labels, flops)
            total_accuracy_sum += accuracy
            total_loss_sum += loss.item()
        avg_loss = total_loss_sum / len(self.test_loader)
        avg_acc = total_accuracy_sum / len(self.test_loader)
        print('Validation:\naccuracy={0:.2f}%, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))
        display_info(flops_list, prob_list)
        # print('acc= {0:.2f}, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))

    # def debug(self, n_epoch=100, load_path='./models/warmup', save_path='./models/search'):
    #     self.load_model(load_path)
    #     tau = 1
    #     total_iter = n_epoch * len(self.train_loader)
    #     current_iter = 0
    #     self.test(tau=tau)
    #
    #     print(self.forward.conv0.conv.weight.shape)
    #     print(self.forward.conv0.gate)
    #     print(self.forward.conv0.conv.weight[:, 0, 0, 0])
    #     # print(self.forward.block_list[0][0].bn0)
    #     # print(self.forward.block_list[0][0].bn1)
    #
    #     data = next(iter(self.test_loader))
    #
    #     self.forward.eval()
    #     inputs, labels = data[0].to(self.device), data[1].to(self.device)
    #     outputs, prob_list, flops, flops_list, conv0, x_bns = self.forward(inputs, tau=tau, noise=False)
    #     accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
    #
    #     conv0_0 = conv0[0,...]
    #     conv0_0_bn = x_bns[0][0,...]
    #
    #     print(conv0_0.shape)
    #     print('======')
    #     print(conv0_0[:, 2, 10])
    #     print(conv0_0_bn[:, 2, 10])
    #     print('-----')
    #
    #     print('======')
    #     print(conv0_0[:, 8, 3])
    #     print(conv0_0_bn[:, 8, 3])
    #     print('-----')
    #
    #
    #     conv0_1 = conv0[1,...]
    #     conv0_1_bn = x_bns[0][1,...]
    #
    #     print('======')
    #     print(conv0_1[:, 2, 10])
    #     print(conv0_1_bn[:, 2, 10])
    #     print('-----')
    #
    #     print('======')
    #     print(conv0_1[:, 8, 3])
    #     print(conv0_1_bn[:, 8, 3])
    #
    #     print(self.forward.block_list[0][0].bn0.weight)
    #     print(self.forward.block_list[0][0].bn0.bias)
    #
    #     print('-----')
    #
    #     '''
    #
    #     As can be observed, the batch norm after the convolution will adjust the outputs of the conv,
    #     using
    #
    #     '''
    #
    #     exit(1)
    #
    #     for epoch in range(n_epoch):
    #         time_prev = timer()
    #         # tau = 10 ** (1 - 2 * epoch / (n_epoch - 1))
    #         print('epoch: ', epoch + 1, ' tau: ', tau)
    #         self.recoder.init({'loss': 0, 'loss_f': 0, 'accuracy': 0,
    #                            'lr': self.opt_train.param_groups[0]['lr'],
    #                            'tau': tau})
    #         for i, data in enumerate(self.train_loader):
    #             # tau = 10 ** (1 - 2 * current_iter / (total_iter-1))
    #             current_iter += 1
    #
    #             # optimizing weights with training data
    #             inputs, labels = data[0].to(self.device), data[1].to(self.device)
    #             self.net.train()
    #             outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
    #             accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
    #             self.opt_train.zero_grad()
    #             loss.backward()
    #             self.opt_train.step()
    #
    #             # optimizing gates with searching data
    #             inputs, labels = data[2].to(self.device), data[3].to(self.device)
    #             self.net.eval()
    #             outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
    #             accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops, prob_list)
    #             self.opt_search.zero_grad()
    #             loss_with_flops.backward()
    #             self.opt_search.step()
    #
    #             self.recoder.add_info(labels.size(0), {'loss': loss, 'loss_f': loss_with_flops,
    #                                                    'accuracy': accuracy})
    #             if (i + 1) % 100 == 0:
    #                 '''
    #                 # for prob in prob_list:
    #                 #     for item in prob.tolist():
    #                 #         print('{0:.2f}'.format(item), end=' ')
    #                 #     print()
    #                 #     break
    #                 # for item in self.forward.named_parameters():
    #                 #     if '0.0.bn0' in item[0] and 'bias' not in item[0]:
    #                 #         print(item)
    #                 '''
    #                 self.net.eval()
    #                 inputs, labels = data[2].to(self.device), data[3].to(self.device)
    #                 outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
    #                 accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
    #                 time_step = timer() - time_prev
    #                 speed = int(100 * self.batch_size_train / time_step)
    #                 print(i + 1,
    #                       ': lr={0:.1e} | acc={1:5.2f} |'.format(self.opt_train.param_groups[0]['lr'], accuracy * 100),
    #                       'loss={0:5.2f} | loss_f={1:5.2f} | flops={2} | speed={3} pic/s'.format(
    #                           loss, loss_with_flops, flops, speed))
    #                 time_prev = timer()
    #                 # self.test(tau=tau)
    #         self.recoder.update(epoch)
    #         self.lr_scheduler_train.step()
    #         self.lr_scheduler_search.step() # adam optimizer do not use multi-step learning rate
    #         if (epoch + 1) % 10 == 0:
    #             self.save_model(os.path.join(save_path, 'model_' + str(epoch + 1) + '.pth'))
    #             self.test(tau=tau)
    #
    #     print('Finished Training')
    #     return tau


def get_prune_list(resnet_channel_list, prob_list, dcfg, expand_rate=0.0001):
    import numpy as np
    prune_list = []
    idx = 0

    chn_input_full, chn_output_full = 3, resnet_channel_list[0]
    dnas_conv = lambda input, output: DNAS.Conv2d(input, output, 1, 1, 1, False, dcfg=dcfg)
    conv = dnas_conv(chn_input_full, chn_output_full)
    chn_output_prune = int(np.round(
        min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
    ))
    chn_output_prune += int(np.ceil(expand_rate*(chn_output_full-chn_output_prune)))
    prune_list.append(chn_output_prune)
    chn_input_full = chn_output_full
    idx += 1
    for blocks in resnet_channel_list[1:]:
        blocks_list = []
        for block in blocks:
            block_prune_list = []
            for chn_output_full in block:
                conv = DNAS.Conv2d(chn_input_full, chn_output_full, 1, 1, 1, False, dcfg=dcfg)
                print(prob_list[idx], conv.out_plane_list, torch.dot(prob_list[idx], conv.out_plane_list).item())
                chn_output_prune = int(np.round(
                    min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
                ))
                chn_output_prune += int(np.ceil(expand_rate*(chn_output_full-chn_output_prune)))
                block_prune_list.append(chn_output_prune)
                chn_input_full = chn_output_full
                idx += 1
            blocks_list.append(block_prune_list)
        prune_list.append(blocks_list)
    return prune_list


def display_info(flops_list, prob_list):
    print('=============')
    for flops, prob in zip(flops_list, prob_list):
        pinfo = ''
        for item in prob.tolist():
            pinfo += '{0:.3f} '.format(item)
        pinfo += ', {0:.0f}'.format(flops)
        print(pinfo)
    print('-------------\n')
