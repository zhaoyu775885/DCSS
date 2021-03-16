import os
import sys
import argparse
from datasets.div2k import DIV2K
from nets.EDSR_gated import EDSR, EDSRDcps
from learner.prunesr import DcpsSRLearner
from learner.fullsr import FullSRLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIV2K', choices=['DIV2K'], help='Dataset Name')
    parser.add_argument('--data_path', default='/home/zhaoyu/Datasets/DIV2K/', type=str, help='Dataset Directory')
    parser.add_argument('--net', default='EDSR', help='Net')
    parser.add_argument('--scale', default=4, type=int, help='scale')
    parser.add_argument('--num_epoch', default=120, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--batch_size_test', default=1, type=int, help='Batch Size for Test')
    parser.add_argument('--std_batch_size', default=64, type=int, help='Norm Batch Size')
    parser.add_argument('--std_init_lr', default=2e-3, type=float, help='Norm Init Lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--prune_flag', default=0, type=int, help='Prune Flag')
    parser.add_argument('--weight_flops', default=0.5, type=float, help='Weight of log(FLOPs) is Loss')
    parser.add_argument('--num_epoch_warmup', default=100, type=int, help='Number of Epochs for warmup')
    parser.add_argument('--num_epoch_search', default=100, type=int, help='Number of Epochs for search')
    parser.add_argument('--full_dir', type=str, help='Index')
    parser.add_argument('--log_dir', type=str, help='Index')
    parser.add_argument('--slim_dir', type=str, help='Index')
    parser.add_argument('--warmup_dir', type=str, help='Index')
    parser.add_argument('--search_dir', type=str, help='Index')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    args = parser.parse_args()

    device = 'cuda:0'
    dataset = DIV2K(args.data_path, scale=args.scale, enlarge=False, length=128000)

    if not args.prune_flag:
        net = EDSR(num_blocks=16, num_chls=64, num_colors=3, scale=args.scale, res_scale=1)
        learner = FullSRLearner(dataset, net, device, args)
        learner.train(n_epoch=args.num_epoch, save_path=args.full_dir)
        learner.load_model(args.full_dir)
        learner.test()
    else:
        net = EDSRDcps(num_blocks=16, num_colors=3, scale=args.scale, res_scale=0.1)
        learner = DcpsSRLearner(dataset, net, device, args)
        learner.train(n_epoch=args.num_epoch, save_path=args.slim_dir)


if __name__ == '__main__':
    main()
