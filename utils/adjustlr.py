from math import cos, pi
def adjust_learning_rate(args, init_lr, optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup_flag else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.num_epoch * num_iter

    if args.lr_decay == 'step':
        lr = init_lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = init_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = init_lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = init_lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = init_lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
