import torch
import torchvision.transforms as transforms
from datasets.dataset_sr import DatasetHDF5, TestDataHDF5
import os


class DIV2K():
    def __init__(self, data_dir, scale, enlarge, length):
        self.data_dir = data_dir
        self.data_path_train = os.path.join(self.data_dir, 'SR_X{0}.hdf5'.format(scale))
        self.data_path_test = os.path.join(self.data_dir, 'X{0}Test.hdf5'.format(scale))
        self.scale = scale
        self.enlarge = enlarge
        self.length = length
        self.patch_size_train = 48
        self.patch_size_test = 0

    def build_dataloader(self, batch_size, is_train=True, valid=False, search=False):
        if is_train:
            dataset = DatasetHDF5(h5_path=self.data_path_train, length=self.length, patch_size=self.patch_size_train,
                                  scale=self.scale, enlarge=self.enlarge, search=search)

        else:
            dataset = TestDataHDF5(h5_path=self.data_path_test, patch_size=self.patch_size_test,
                                   scale=self.scale, enlarge=False)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=16,
                                                  pin_memory=True)

        return data_loader


if __name__ == '__main__':
    data_dir = '/home/zhaoyu/Data/DIV2K/'
    dataset = DIV2K(data_dir, scale=2, enlarge=False, length=128000)
    dataloader = dataset.build_dataloader(batch_size=64, is_train=True, search=False)
    for i, item in enumerate(dataloader):
        if i > 1:
            break
        print(len(item))
        print(item[0].shape, item[1].shape)

    dataloader = dataset.build_dataloader(batch_size=64, is_train=False, search=True)
    for i, item in enumerate(dataloader):
        if i > 1:
            break
        print(len(item))
