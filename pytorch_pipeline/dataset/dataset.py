import cv2
from pathlib import Path
from torch.nn import functional as F
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as torch_transforms
from dataset import transforms

from torch.utils import data

class TGSSaltDataset(data.Dataset):
    def get_file_list(self, root_path, file_list, test_file_list):
        result_file_list = []
        train_path = os.path.join(root_path, 'train')
        for file in file_list:
            result_file_list.append(os.path.join(os.path.join(train_path, "images"), file + ".png"))
        return result_file_list + test_file_list

    def __init__(self, root_path, file_list, test_file_list=[], augment=True):
        self.root_path = root_path
        self.file_list = self.get_file_list(root_path=root_path,file_list=file_list, test_file_list=test_file_list)
        self.image_transforms = [
            transforms.Brightness(0.2, prob=0.4),
        ]

        self.dataset_transforms = [
            transforms.Mirror(),
            transforms.Rotate(prob=0.4, min_angle=-10, max_angle=10, )
        ]
        self.torch_transforms = torch_transforms.Compose([
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def get_mask_path(self, image_path):
        path, filename = os.path.split(image_path)
        root_folder, _ = os.path.split(path)
        return os.path.join(root_folder, 'masks', filename)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        image_path = self.file_list[index]

        mask_path = self.get_mask_path(image_path)
        image = cv2.imread(str(image_path))
        image = image / 255.0

        if self.augment:
            for t in self.image_transforms:
                image = t(image)
        mask = cv2.imread(str(mask_path))
        mask = mask / 255.0
        data = np.dstack((image, mask))
        if self.augment:
            for t in self.dataset_transforms:
                data = t(data)
        image = cv2.resize(data[:, :, 0:3], (128, 128), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(data[:, :, 3:], (128, 128), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype('float32'))
        image = self.torch_transforms(image)
        mask = np.rint(mask[:, :, 0:1])
        # visible = np.sum(mask) != 0
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32'))
        result_dict = {'image': image,
                       'mask': mask}  # for possible additional parameters
        return result_dict
