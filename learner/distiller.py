import torch.nn as nn
from learner.full import FullLearner


class Distiller(FullLearner):
    def __init__(self, dataset, net, device, args, model_path):
        super(Distiller, self).__init__(dataset, net, device, args)
        print(device)
        ''' handle exceptions '''
        self.load_model(model_path)
        self.kd_loss_fn = self._setup_kd_loss_fn()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.net.eval()
        print('Performance of the Teacher')
        self.test()

    def _setup_kd_loss_fn(self):
        return nn.KLDivLoss(reduction='batchmean')

    def infer(self, images):
        return self.net(images).detach()

    def kd_loss(self, std_logits, trg_logits):
        t_dst = self.args.dst_temperature
        w_dst = self.args.dst_loss_weight
        log_prob = self.log_softmax(std_logits / t_dst)
        prob = self.softmax(trg_logits / t_dst)
        loss = self.kd_loss_fn(log_prob, prob) * w_dst
        return loss


if __name__ == '__main__':
    from datasets.cifar import Cifar100
    from nets.resnet import ResNet20

    cifar100_path = '/home/zhaoyu/Datasets/cifar100'
    cifar100 = Cifar100(cifar100_path)

    net = ResNet20(n_classes=100)
    learner = Distiller(cifar100, net, device='cpu')

    learner.test()
