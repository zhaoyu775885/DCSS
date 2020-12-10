# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:43:41 2020

@author: enix3
"""

import numpy as np

import h5py
import torch.utils.data as data
import torch
import random
import cv2


# Must with key as 'in_data' and 'gt'
# patch_size is the patch size of lr image
class DatasetHDF5(data.Dataset):
    def __init__(self, h5_path, length=None, patch_size=48, scale=2, enlarge=False):
        super(DatasetHDF5, self).__init__()
        self.length = length
        # self.batch_size = batch_size
        self.patch_size = patch_size
        self.scale = scale
        self.enlarge = enlarge
        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file['in_data'].keys())
            self.num_images = len(self.keys)

    def __getitem__(self, index):
        ind_im = random.randint(0, self.num_images - 1)
        with h5py.File(self.h5_path, 'r') as h5_file:
            indices = self.keys[ind_im]
            im_gt = np.array(h5_file['gt'][indices])
            in_data = np.array(h5_file['in_data'][indices])
        in_data, im_gt = self._get_patch(in_data, im_gt)
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        in_data = torch.from_numpy(in_data.transpose((2, 0, 1)))

        return in_data, im_gt

    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            return self.num_images

    def _get_patch(self, lr, hr):
        h, w, c = lr.shape
        ip = self.patch_size
        tp = ip * self.scale
        ix = random.randrange(0, w - ip + 1)  # LR patch index
        iy = random.randrange(0, h - ip + 1)
        tx = self.scale * ix  # HR patch index
        ty = self.scale * iy
        lr = lr[iy:iy + ip, ix:ix + ip, :].astype(np.float32)
        hr = hr[ty:ty + tp, tx:tx + tp, :].astype(np.float32)
        lr, hr = random_augmentation(lr, hr)
        if self.enlarge:
            lr = cv2.resize(lr, (tp, tp), interpolation=cv2.INTER_CUBIC)
        return lr / 255., hr / 255.


class TestDataHDF5(data.Dataset):
    def __init__(self, h5_path, enlarge=False, scale=2, patch_size=0):
        super(TestDataHDF5, self).__init__()
        self.h5_path = h5_path
        self.scale = scale
        self.enlarge = enlarge
        self.patch_size = patch_size
        with h5py.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file['in_data'].keys())
            self.num_images = len(self.keys)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as h5_file:
            indices = self.keys[index]
            im_gt = np.array(h5_file['gt'][indices])
            in_data = np.array(h5_file['in_data'][indices])

            if self.patch_size > 0:
                in_data, im_gt = self._get_patch(in_data, im_gt)

            if self.enlarge:
                h, w, c = in_data.shape
                in_data = cv2.resize(in_data, (h * self.scale, w * self.scale), interpolation=cv2.INTER_CUBIC)

            im_gt = im_gt.transpose((2, 0, 1)).astype(np.float32)
            in_data = in_data.transpose((2, 0, 1)).astype(np.float32)

        im_gt = torch.from_numpy(im_gt)
        in_data = torch.from_numpy(in_data)

        return in_data / 255., im_gt / 255.

    def _get_patch(self, lr, hr):
        h, w, c = lr.shape
        ip = self.patch_size
        tp = ip * self.scale
        ix = random.randrange(0, w - ip + 1)  # LR patch index
        iy = random.randrange(0, h - ip + 1)
        tx = self.scale * ix  # HR patch index
        ty = self.scale * iy
        lr = lr[iy:iy + ip, ix:ix + ip, :]
        hr = hr[ty:ty + tp, tx:tx + tp, :]
        lr, hr = random_augmentation(lr, hr)
        if self.enlarge:
            lr = cv2.resize(lr, (tp, tp), interpolation=cv2.INTER_CUBIC)
        return lr, hr


def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        pass
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out


def random_augmentation(img1, img2):
    # if random.randint(0,1) == 1:
    flag_aug = random.randint(0, 7)
    if flag_aug > 0:
        img1 = data_augmentation(img1, flag_aug)
        img2 = data_augmentation(img2, flag_aug)
    return img1, img2
