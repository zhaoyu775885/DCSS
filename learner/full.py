import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from learner.abstract_learner import AbstractLearner
import os
from utils.writer import reduce_mean


class FullLearner(AbstractLearner):
    def __init__(self, dataset, net, args, teacher=None):
        super(FullLearner, self).__init__(dataset, net, args)

        self.forward = self.net if torch.cuda.device_count() == 1 else \
            torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[args.local_rank])

        self.batch_size_train = self.args.batch_size
        self.batch_size_test = self.args.batch_size_test
        self.train_loader = self._build_dataloader(self.batch_size_train, is_train=True)
        self.test_loader = self._build_dataloader(self.batch_size_test, is_train=False)

        print(self.args.batch_size, self.args.std_batch_size, self.args.std_init_lr)
        self.init_lr = self.batch_size_train / self.args.std_batch_size * self.args.std_init_lr * self.args.nproc
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

        self.teacher = teacher

    def _setup_loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        return optim.SGD(self.forward.parameters(), lr=self.init_lr,
                         momentum=self.args.momentum, weight_decay=self.args.weight_decay)

    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.args.num_epoch)

    def metrics(self, logits, labels, trg_logits=None):
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum()
        loss = self.loss_fn(logits, labels)
        accuracy = correct / labels.size(0)
        kd_loss = 0
        if trg_logits is not None:
            kd_loss = self.teacher.kd_loss(logits, trg_logits)
            return accuracy, loss+kd_loss, kd_loss
        return accuracy, loss + kd_loss

    def cnt_flops(self):
        flops = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
            flops = self.forward.cnt_flops(inputs)
            break
        return flops

    def train(self, n_epoch, save_path='./models/full/model.pth'):
        self.forward.train()
        print(n_epoch)
        for epoch in range(n_epoch):
            if self.args.local_rank == 0:
                print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'accuracy': 0, 'lr': self.opt.param_groups[0]['lr']})

            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
                logits = self.forward(inputs)
                if self.teacher is None:
                    accuracy, loss = self.metrics(logits, labels)
                else:
                    trg_logits = self.teacher.infer(inputs)
                    accuracy, loss, kd_loss = self.metrics(logits, labels, trg_logits)
                self.recoder.add_info(labels.size(0), {'loss': loss, 'accuracy': accuracy})

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (i + 1) % self.args.print_steps == 0 and self.args.local_rank == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * self.batch_size_train / time_step) * self.args.nproc
                    print(i + 1, ': lr={0:.1e} | acc={1:5.2f} | loss={2:5.2f} | speed={3} pic/s'.format(
                        self.opt.param_groups[0]['lr'], accuracy * 100, loss, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler.step()

            if (epoch + 1) % self.args.save_epochs == 0:
                self.test()
                self.forward.train()
                if self.args.local_rank == 0:
                    self.save_model(os.path.join(save_path, 'model_'+str(epoch+1)+'.pth'))

        print('Finished Training')

    def test(self):
        self.forward.eval()
        total_accuracy_sum, total_loss_sum = 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                inputs, labels = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
                logits = self.forward(inputs)
                accuracy, loss = self.metrics(logits, labels)

                if self.args.nproc > 1:
                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(loss, self.args.nproc)
                    reduced_accuracy = reduce_mean(accuracy, self.args.nproc)
                else:
                    reduced_loss, reduced_accuracy = loss, accuracy

                total_accuracy_sum += reduced_accuracy
                total_loss_sum += reduced_loss

        if self.args.local_rank == 0:
            avg_loss = total_loss_sum / len(self.test_loader)
            avg_acc = total_accuracy_sum / len(self.test_loader)
            print('Validation Performance:\naccuracy={0:.2f}%, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))

        torch.cuda.empty_cache()
