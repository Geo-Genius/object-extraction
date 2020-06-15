import os
import numpy as np
from PIL import Image

import torch
from .base import BaseDataset

class NormSegmentation(BaseDataset):
    NUM_CLASS = 6
    def __init__(self, root, split='train', mode=None, transform=None,
                 target_transform=None, **kwargs):
        super(NormSegmentation, self).__init__(root, split, mode, transform,
                                              target_transform, **kwargs)
        self.images = []
        self.masks = []
        if self.mode == 'train':
            f = open(os.path.join(root, 'train.txt'), 'r')
        elif self.mode == 'val':
            f = open(os.path.join(root, 'val.txt'), 'r')
        elif self.mode == 'test':
            f = open(os.path.join(root, 'test.txt'), 'r')
        f_list = f.read().split('\n')[:-1]
        self.images = [os.path.join(root, item.split(' ')[0]) for item in f_list]
        self.masks = [os.path.join(root, item.split(' ')[1]) for item in f_list]
        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            target = self.target_transform(target)
        return img, target

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)
