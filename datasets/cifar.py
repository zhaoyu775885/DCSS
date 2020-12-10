import torch
import torchvision
import torchvision.transforms as transforms
import datasets.search_cifar as dataset_search


class Cifar():
    def __init__(self, data_dir, dataset_fn_base, dataset_fn_search):
        self.dataset_fn_base = dataset_fn_base
        self.dataset_fn_search = dataset_fn_search
        self.data_dir = data_dir

    def build_dataloader(self, batch_size, is_train=True, valid=False, search=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])

        valid_transform = transforms.Compose([transforms.ToTensor(), normalize])

        dataset_fn = self.dataset_fn_search if search else self.dataset_fn_base
        cifar_dataset = dataset_fn(root=self.data_dir, train=is_train,
                                   transform=train_transform if is_train else valid_transform)

        print('---------')
        print(len(cifar_dataset))

        if valid:
            # todo: redundancy
            # 代码实现细节，为什么抛弃这种构造search数据集合的方式
            # 最初判断为无法在val loader的多次循环中实现不同的shuffle，待确认
            dataset_train, dataset_valid = torch.utils.data.random_split(cifar_dataset, [45000, 5000])
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, drop_last=True,
                                                       shuffle=is_train, num_workers=16)
            valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, drop_last=True,
                                                       shuffle=is_train, num_workers=16)
            return train_loader, valid_loader

        return torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, shuffle=is_train, num_workers=16)


class Cifar10(Cifar):
    def __init__(self, data_dir):
        super(Cifar10, self).__init__(data_dir,
                                      dataset_fn_base=torchvision.datasets.CIFAR10,
                                      dataset_fn_search=dataset_search.CIFAR10Search)
        self.n_class = 10


class Cifar100(Cifar):
    def __init__(self, data_dir):
        super(Cifar100, self).__init__(data_dir,
                                       dataset_fn_base=torchvision.datasets.CIFAR100,
                                       dataset_fn_search=dataset_search.CIFAR100Search)
        self.n_class = 100


if __name__ == '__main__':
    dataset = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(dataset)

    train = dataset.build_dataloader(128, search=True)

    for i, batch in enumerate(train):
        print(i, batch[1])

# def imshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# def instance(dataloader, classes, batch_size=BATCH_SIZE):
#     dataiter = iter(dataloader)
#     images, labels = dataiter.next()
#     imshow(torchvision.utils.make_grid(images))
#     print(' '.join([classes[labels[j]] for j in range(batch_size)]))
