import os
import sys
import argparse
import torch
from datasets.cifar import Cifar10, Cifar100
from datasets.imagenet import ImageNet
from nets.resnet import ResNet
from nets.mobilenetv2 import mobilenet_v2
from nets.resnet_lite import ResNetLite
from nets.resnet_gated import ResNetGated
from learner.prune import DcpsLearner
from learner.full import FullLearner
from learner.distiller import Distiller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset Name')
    parser.add_argument('--data_path', type=str, help='Dataset Directory')
    parser.add_argument('--net', default='resnet', choices=['resnet', 'mobilenet'], help='Net')
    parser.add_argument('--net_index', type=int, choices=[1, 2, 18, 20, 32, 34, 50, 56], help='Index')
    parser.add_argument('--num_epoch', default=250, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--batch_size_test', default=100, type=int, help='Batch Size for Test')
    parser.add_argument('--std_batch_size', default=128, type=int, help='Norm Batch Size')
    parser.add_argument('--std_init_lr', default=1e-1, type=float, help='Norm Init Lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--dst_flag', default=0, type=int, help='Dst Flag')
    parser.add_argument('--prune_flag', default=0, type=int, help='Prune Flag')
    parser.add_argument('--weight_flops', default=0.5, type=float, help='Weight of log(FLOPs) is Loss')
    parser.add_argument('--num_epoch_warmup', default=100, type=int, help='Number of Epochs for warmup')
    parser.add_argument('--num_epoch_search', default=100, type=int, help='Number of Epochs for search')
    parser.add_argument('--teacher_net', default='resnet', choices=['resnet'], help='Net')
    parser.add_argument('--teacher_net_index', type=int, choices=[18, 20, 32, 34, 50, 56], help='Index')
    parser.add_argument('--dst_temperature', default=1.0, type=float, help='temperature')
    parser.add_argument('--dst_loss_weight', default=1.0, type=float, help='weight of distillation')
    parser.add_argument('--full_dir', type=str, help='Index')
    parser.add_argument('--log_dir', type=str, help='Index')
    parser.add_argument('--slim_dir', type=str, help='Index')
    parser.add_argument('--warmup_dir', type=str, help='Index')
    parser.add_argument('--search_dir', type=str, help='Index')
    parser.add_argument('--teacher_dir', type=str, help='Index')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--nproc', default=1, type=int, help='number of processes')
    parser.add_argument('--save_epochs', default=10, type=int, help='save checkpoint every "save_epochs"')
    parser.add_argument('--print_steps', default=100, type=int, help='print training info every "print_steps"')

    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        if args.local_rank == 0:
            print('init pytorch distributed parallelism')
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    if args.dataset == 'cifar10':
        dataset_fn = Cifar10
    elif args.dataset == 'cifar100':
        dataset_fn = Cifar100
    elif args.dataset == 'imagenet':
        dataset_fn = ImageNet

    dataset = dataset_fn(args.data_path)
    n_class = dataset.n_class

    teacher = None
    if args.dst_flag:
        teacher_net = ResNet(args.teacher_net_index, n_class)
        teacher = Distiller(dataset, teacher_net, args, model_path=args.teacher_dir)

    if not args.prune_flag:
        if args.net == 'mobilenet':
            net = mobilenet_v2()
        else:
            net = ResNet(args.net_index, n_class)
        learner = FullLearner(dataset, net, args, teacher=teacher)
        learner.train(n_epoch=args.num_epoch, save_path=args.full_dir)
        #learner.load_model(args.full_dir)
        #learner.test()
    else:
        net = ResNetGated(args.net_index, n_class)
        learner = DcpsLearner(dataset, net, args, teacher=teacher)
        learner.train(n_epoch=args.num_epoch, save_path=args.slim_dir)


if __name__ == '__main__':
    main()
