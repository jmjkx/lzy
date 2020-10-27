# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:16:11 2019

@author: viryl
"""

from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from ipdb import set_trace
from torch.utils.data import Dataset as BaseDataset
# 加载输入

class MyDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_npy,
            label_npy,

    ):
        self.images = images_npy
        self.labels = label_npy
        self.length = images_npy.shape[0]
        # convert str names to class values on masks

    def __getitem__(self, i):
        # a= MyDataset(img, label), a[0]
        # read data
        image = self.images[i]

        label = self.labels[i]

        return torch.from_numpy(image), torch.from_numpy(np.array(label))

    def __len__(self):
        return self.length

# 返回一个patch 和label, a= MyDataset(img, label), len(a)


# 包含patch image和相应label的元组

