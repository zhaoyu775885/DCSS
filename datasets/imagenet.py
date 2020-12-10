import torch
import torchvision
import torchvision.transforms as transforms
import datasets.search_imagenet as dataset_search


class ImageNet:
    def __init__(self, data_dir):
        self.dataset_fn_base = torchvision.datasets.ImageNet
        self.dataset_fn_search = dataset_search.ImageNetSearch
        self.data_dir = data_dir
        self.n_class = 1000

    def build_dataloader(self, batch_size, is_train=True, valid=False, search=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.RandomCrop(224),
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

        print('---------')
        print(len(imagenet_dataset))

        # if valid:
        #     dataset_train, dataset_valid = torch.utils.data.random_split(cifar_dataset, [45000, 5000])
        #     train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, drop_last=True,
        #                                                shuffle=is_train, num_workers=16)
        #     valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, drop_last=True,
        #                                                shuffle=is_train, num_workers=16)
        #     return train_loader, valid_loader

        return torch.utils.data.DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=is_train, num_workers=16, pin_memory=True)


if __name__ == '__main__':
    dataset = '/home/zhaoyu/Datasets/Imagenet2012'
    dataset = ImageNet(dataset)

    train = dataset.build_dataloader(128, search=True)

    for i, item in enumerate(train):
        print(i)
        print(item[0].shape, item[1])
        break
