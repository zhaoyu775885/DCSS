import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from nets.resnet_lite import ResNetL
from learner.abstract_learner import AbstractLearner
from learner.full import FullLearner
import utils.DNAS as DNAS
from utils.DNAS import entropy
import os
from nets.resnet_lite import ResNetChannelList


class DcpsLearner(AbstractLearner):
    def __init__(self, dataset, net, args, teacher=None):
        super(DcpsLearner, self).__init__(dataset, net, args)

        self.forward = self.net

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
        if torch.cuda.device_count() > 1:
            self.forward = nn.DataParallel(self.forward, device_ids=[0, 1])

    def _setup_loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        pass

    def _setup_optimizer_warmup(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=self.init_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def _setup_optimizer_train(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=self.init_lr * 0.1, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def _setup_optimizer_search(self):
        gates = [item[1] for item in self.forward.named_parameters() if 'gate' in item[0]]
        return optim.Adam(gates, lr=self.init_lr * 0.1)

    def _setup_lr_scheduler_warmup(self):
        if self.args.dataset == 'imagenet':
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[30, 50], gamma=0.1)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[100, 150], gamma=0.1)

    def _setup_lr_scheduler_train(self):
        if self.args.dataset == 'imagenet':
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[30, 50], gamma=0.1)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[50, 100], gamma=0.1)

    def _setup_lr_scheduler_search(self):
        if self.args.dataset == 'imagenet':
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[30, 50], gamma=0.1)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[50, 100], gamma=0.1)

    def metrics(self, outputs, labels, flops=None, prob_list=None):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.loss_fn(outputs, labels)
        prob_loss = 0
        if prob_list is not None:
            for prob in prob_list:
                prob_loss += entropy(prob)
            loss += 0.00 * prob_loss
        coef = self.args.weight_flops
        loss_with_flops = loss + coef * torch.log(flops)
        accuracy = correct / labels.size(0)
        return accuracy, loss, loss_with_flops

    def train(self, n_epoch=250, save_path='./models/slim'):
        #self.train_warmup(n_epoch=self.args.num_epoch_warmup, save_path=self.args.warmup_dir)
        #tau = self.train_search(n_epoch=self.args.num_epoch_search,
        #                        load_path=self.args.warmup_dir,
        #                        save_path=self.args.search_dir)
        tau = 0.1
        self.train_prune(tau=tau, n_epoch=n_epoch,
                         load_path=self.args.search_dir,
                         save_path=save_path)

    def squeeze(self, data):
        extract_data = []
        length = int(data[0].size(0) // 2)
        for item in data[:]:
            extract_data.append(item[:length] if length > 2 else item[0])
        return extract_data

    def train_warmup(self, n_epoch=200, save_path='./models/warmup'):
        print('Warmup', n_epoch, 'epochs')
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'accuracy': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs, _, flops, flops_list = self.forward(inputs, tau=1.0, noise=False)
                if torch.cuda.device_count() > 1:
                    flops, flops_list = flops[0], self.squeeze(flops_list)
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
                self.save_model(os.path.join(save_path, 'model_' + str(epoch + 1) + '.pth'))
                self.test(tau=1.0)
                self.net.train()
        print('Finished Warming-up')

    def train_search(self, n_epoch=100, load_path='./models/warmup', save_path='./models/search'):
        print('Search', n_epoch, 'epochs')
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
                tau = 10 ** (1 - 2.0 * current_iter / (total_iter - 1))
                current_iter += 1

                # optimizing weights with training data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.net.train()
                outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=True)
                if torch.cuda.device_count() > 1:
                    flops, prob_list, flops_list = flops[0], self.squeeze(prob_list), self.squeeze(flops_list)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.opt_train.zero_grad()
                loss.backward()
                self.opt_train.step()

                # optimizing gates with searching data
                inputs, labels = data[2].to(self.device), data[3].to(self.device)
                self.net.eval()
                outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
                if torch.cuda.device_count() > 1:
                    flops, prob_list, flops_list = flops[0], self.squeeze(prob_list), self.squeeze(flops_list)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.opt_search.zero_grad()
                loss_with_flops.backward()
                self.opt_search.step()

                self.recoder.add_info(labels.size(0), {'loss': loss, 'loss_f': loss_with_flops,
                                                       'accuracy': accuracy})
                if (i + 1) % 100 == 0:
                    # todo: to be deleted
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
            if (epoch + 1) % 5 == 0:
                self.test(tau=tau)
                self.save_model(os.path.join(save_path, 'model_' + str(epoch + 1) + '.pth'))

        print('Finished Training')
        return tau

    def train_prune(self, tau, n_epoch=250,
                    load_path='./models/search/model.pth',
                    save_path='./models/prune/model.pth'):
        print('Train', n_epoch, 'epochs')
        # Done, 0. load the searched model and extract the prune info
        # Done, 1. define the slim network based on prune info
        # Done, 2. train and validate, and exploit the full learner
        self.load_model(load_path)
        dcfg = DNAS.DcpConfig(n_param=8, split_type=DNAS.TYPE_A, reuse_gate=None)
        channel_list = ResNetChannelList(self.args.net_index)

        self.net.eval()
        data = next(iter(self.train_loader))
        inputs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
        outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
        if torch.cuda.device_count() > 1:
            flops, prob_list, flops_list = flops[0], self.squeeze(prob_list), self.squeeze(flops_list)
        display_info(flops_list, prob_list)
        channel_list_prune = get_prune_list(channel_list, prob_list, dcfg=dcfg)
        """
        channel_list_prune = [43, [[58, 32, 231, 231], [44, 58, 231], [52, 64, 231]],
                [[72, 84, 359, 359], [64, 103, 359], [72, 116, 359], [83, 72, 359]],
                [[205, 141, 820, 820], [134, 154, 820], [124, 154, 820], [142, 119, 820], [143, 200, 820], [149, 147, 820]],
                [[360, 328, 1844, 1844], [327, 377, 1844], [359, 314, 1844]]]
        """
        print(channel_list_prune)
        del data, inputs, labels, outputs, prob_list, flops, flops_list
        

        net = ResNetL(self.args.net_index, self.dataset.n_class, channel_list_prune)
        full_learner = FullLearner(self.dataset, net, args=self.args, teacher=self.teacher)
        if torch.cuda.device_count() == 1:
            # only work for single GPU mode
            print('FLOPs:', full_learner.cnt_flops())
        full_learner.train(n_epoch=n_epoch, save_path=save_path)

    def test(self, tau=1.0):
        self.net.eval()
        total_accuracy_sum, total_loss_sum = 0, 0
        flops_list, prob_list = [], []
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
                outputs, prob_list, flops, flops_list = self.forward(images, tau=tau, noise=False)
                if torch.cuda.device_count() > 1:
                    flops, prob_list, flops_list = flops[0], self.squeeze(prob_list), self.squeeze(flops_list)
                # todo: to be fixed
                accuracy, loss, _ = self.metrics(outputs, labels, flops)
                total_accuracy_sum += accuracy
                total_loss_sum += loss.item()

        avg_loss = total_loss_sum / len(self.test_loader)
        avg_acc = total_accuracy_sum / len(self.test_loader)
        print('Validation:\naccuracy={0:.2f}%, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))
        display_info(flops_list, prob_list)
        torch.cuda.empty_cache()


def get_prune_list(resnet_channel_list, prob_list, dcfg, expand_rate: float = 0):
    import numpy as np
    prune_list = []
    idx = 0

    chn_input_full, chn_output_full = 3, resnet_channel_list[0]
    dnas_conv = lambda input, output: DNAS.Conv2d(input, output, 1, 1, 1, False, dcfg=dcfg)
    conv = dnas_conv(chn_input_full, chn_output_full)
    chn_output_prune = int(np.round(
        min(torch.dot(prob_list[idx].to('cpu'), conv.out_plane_list).item(), chn_output_full)
    ))
    chn_output_prune += int(np.ceil(expand_rate * (chn_output_full - chn_output_prune)))
    prune_list.append(chn_output_prune)
    chn_input_full = chn_output_full
    idx += 1
    for blocks in resnet_channel_list[1:]:
        blocks_list = []
        for block in blocks:
            block_prune_list = []
            for chn_output_full in block:
                conv = DNAS.Conv2d(chn_input_full, chn_output_full, 1, 1, 1, False, dcfg=dcfg)
                print(idx, prob_list[idx].to('cpu'), conv.out_plane_list, torch.dot(prob_list[idx].to('cpu'), conv.out_plane_list).item())
                chn_output_prune = int(np.round(
                    min(torch.dot(prob_list[idx].to('cpu'), conv.out_plane_list).item(), chn_output_full)
                ))
                chn_output_prune += int(np.ceil(expand_rate * (chn_output_full - chn_output_prune)))
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
