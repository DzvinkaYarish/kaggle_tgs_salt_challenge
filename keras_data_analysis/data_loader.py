from keras.preprocessing.image import load_img
import os
import pandas as pd
import numpy as np


class DataLoader():
    def __init__(self, csv_file, root_dir, transform=None, mask=True):
        self.imgs_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.imgs_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images',
                                self.imgs_frame.iloc[idx, 0])
        image = (np.array(load_img(img_name + '.png',  grayscale=True))) / 255
        sample = {'img': image}
        if self.mask:
            mask_name = os.path.join(self.root_dir, 'masks',
                                    self.imgs_frame.iloc[idx, 0])
            mask = (np.array(load_img(mask_name + '.png', grayscale=True))) / 255
            sample = {'img': image, 'mask': mask}

        if self.transform:
            if self.mask:
                sample = self.transform(image=image, mask=mask)
            else:
                sample = self.transform(image=image)

        return sample
