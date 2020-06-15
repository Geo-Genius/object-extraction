import os
import numpy as np
from PIL import Image
import cv2
import torch
from .base import BaseDataset


class PostamSegmentation(BaseDataset):
    BASE_DIR = 'postam'
    NUM_CLASS = 6
    STEP = 256

    def __init__(self, root='../datasets', split='train', img_patch_set=None, mask_patch_set=None, index_list=None,
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(PostamSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.img_patch_set = img_patch_set
        self.mask_patch_set = mask_patch_set
        self.index_list = index_list
        self.rows, self.cols = img_patch_set.shape


    def __getitem__(self, index):
        index = self.index_list[index]
        h = index // self.cols
        w = index % self.cols
        img = Image.fromarray(self.img_patch_set.patch_index[h][w].data['BANDS'][:, :, :, 0:3].squeeze()).convert('RGB')
        mask_arr = self.mask_patch_set.patch_index[h][w].data['BANDS'][:, :, :, ].squeeze()

        mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_RGB2GRAY)
        mask_arr[mask_arr == 0] = 0
        mask_arr[mask_arr == 255] = 0
        mask_arr[mask_arr == 76] = 1
        mask_arr[mask_arr == 226] = 2
        mask_arr[mask_arr == 150] = 3
        mask_arr[mask_arr == 179] = 4
        mask_arr[mask_arr == 29] = 5
        if (mask_arr > 5).any():
            print("error")
        mask = Image.fromarray(mask_arr)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.index_list)
