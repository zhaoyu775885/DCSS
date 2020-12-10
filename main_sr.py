import os
import sys
import argparse
from datasets.cifar import Cifar10, Cifar100
from datasets.div2k import DIV2K
from nets.EDSR_gated import EDSR, EDSRLite
from nets.resnet_gated import ResNetGated
from learner.prune import DcpsLearner
from learner.fullsr import FullSRLearner
from learner.distiller import Distiller

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# prepare for parallel training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIV2K', choices=['DIV2K'], help='Dataset Name')
    # parser.add_argument('--data_path', default='/home/zhaoyu/Datasets/DIV2K/', type=str, help='Dataset Directory')
    parser.add_argument('--data_path', default='/home/zhaoyu/Projects/python/SR/SR_0921/Datasets', type=str, help='Dataset Directory')
    parser.add_argument('--net', default='resnet', choices=['resnet', 'mobilenet'], help='Net')
    parser.add_argument('--net_index', default=20, type=int, choices=[20, 32, 56], help='Index')
    parser.add_argument('--num_epoch', default=120, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--batch_size_test', default=1, type=int, help='Batch Size for Test')
    parser.add_argument('--std_batch_size', default=64, type=int, help='Norm Batch Size')
    parser.add_argument('--std_init_lr', default=2e-3, type=float, help='Norm Init Lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--dst_flag', default=0, type=int, help='Dst Flag')
    parser.add_argument('--prune_flag', default=0, type=int, help='Prune Flag')
    parser.add_argument('--teacher_net', default='resnet', choices=['resnet'], help='Net')
    parser.add_argument('--teacher_net_index', default=20, type=int, choices=[20, 32, 56], help='Index')
    parser.add_argument('--dst_temperature', default=1, type=float, help='temperature')
    parser.add_argument('--dst_loss_weight', default=1, type=float, help='weight of distillation')
    parser.add_argument('--full_dir', type=str, help='Index')
    parser.add_argument('--log_dir', type=str, help='Index')
    parser.add_argument('--slim_dir', type=str, help='Index')
    parser.add_argument('--warmup_dir', type=str, help='Index')
    parser.add_argument('--search_dir', type=str, help='Index')
    parser.add_argument('--teacher_dir', type=str, help='Index')
    args = parser.parse_args()

    device = 'cuda:0'
    dataset = DIV2K(args.data_path, scale=2, enlarge=False, length=128000)
    net = EDSR(num_blocks=16, num_chls=64, num_color=3, scale=2, res_scale=0.1)
    learner = FullSRLearner(dataset, net, device, args)
    learner.train(n_epoch=args.num_epoch, save_path='workdir/EDSR/full')

    # device = 'cuda:0'
    # teacher = None
    # if args.dst_flag:
    #     teacher_net = ResNet(args.teacher_net_index, n_class)
    #     teacher = Distiller(dataset, teacher_net, device, args, model_path=args.teacher_dir)
    #
    # if not args.prune_flag:
    #     net = ResNet(args.net_index, n_class)
    #     learner = FullLearner(dataset, net, device, args, teacher=teacher)
    #     learner.train(n_epoch=args.num_epoch, save_path=args.full_dir)
    #     learner.load_model(args.full_dir)
    #     learner.test()
    # else:
    #     net = ResNetGated(args.net_index, n_class)
    #     learner = DcpsLearner(dataset, net, device, args, teacher=teacher)
    #     learner.train(n_epoch=args.num_epoch, save_path=args.slim_dir)


if __name__ == '__main__':
    main()
