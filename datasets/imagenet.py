import os
import torch
import torchvision
import torchvision.transforms as transforms
import datasets.search_imagenet as dataset_search
from timeit import default_timer as timer


class ImageNet:
    def __init__(self, data_dir):
        self.dataset_fn_base = torchvision.datasets.ImageNet
        self.dataset_fn_search = dataset_search.ImageNetSearch
        self.data_dir = data_dir
        self.n_class = 1000
        self.sampler = None

    def build_dataloader(self, batch_size, is_train=True, valid=False, search=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        valid_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])

        dataset_fn = self.dataset_fn_search if search else self.dataset_fn_base
        imagenet_dataset = dataset_fn(root=self.data_dir, split='train' if is_train else 'val',
                                      transform=train_transform if is_train else valid_transform)

        print('train' if is_train else 'valid', 'samples: ', len(imagenet_dataset))

        if torch.cuda.device_count() > 1:
            print('distributed parallelism')
            self.sampler = torch.utils.data.distributed.DistributedSampler(imagenet_dataset, shuffle=is_train)
            return torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=(self.sampler is None), drop_last=False,
                                               sampler=self.sampler, pin_memory=True, num_workers=4*torch.cuda.device_count())

        return torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=is_train, drop_last=False,
                                           pin_memory=True, num_workers=4*torch.cuda.device_count())

    def get_sampler(self):
        if self.sampler is None:
            print('ERROR! Sampler is not initialized')
        return self.sampler

"""
class ImageNet:
    def __init__(self, data_dir):
        self.dataset_fn_base = torchvision.datasets.ImageNet
        self.data_dir = data_dir
        self.n_class = 1000

    def build_dataloader(self, batch_size, is_train=True, valid=False, search=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        valid_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])

        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(self.data_dir, 'train' if is_train else 'val'),
            transform=train_transform if is_train else valid_transform)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   sampler=train_sampler)

        if torch.cuda.device_count() > 1:
            return train_loader

        return train_loader
"""


if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')

    dataset = '/home/zhaoyu/Data/Imagenet/ILSVRC2012'
    dataset = ImageNet(dataset)

    train_dataloader = dataset.build_dataloader(256)

    time_prev = timer()
    for i, (images, target) in enumerate(train_dataloader):
        if (i+1) % 10 == 0:
            time_step = timer() - time_prev
            print(i+1, ':', time_step)
            time_prev = timer()

