import torch
import torch.utils.data
import numpy as np
import cv2
import os

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, train_data_path, train_mask_path, transform=None):
        self.train_data_path = road_segmentation_dataset/train/images
        self.train_mask_path = road_segmentation_dataset/train/images
        self.transform = transform

        self.examples = []
        file_names = os.listdir(self.train_data_path)
        for file_name in file_names:
            img_id = file_name.split(".")[0]
            img_path = os.path.join(self.train_data_path, file_name)
            mask_path = os.path.join(self.train_mask_path, img_id + "_mask.png")
            self.examples.append((img_path, mask_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, mask_path = self.examples[index]
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, val_data_path, val_mask_path, transform=None):
        self.val_data_path = road_segmentation_dataset/val/images
        self.val_mask_path = road_segmentation_dataset/val/images
        self.transform = transform

        self.examples = []
        file_names = os.listdir(self.val_data_path)
        for file_name in file_names:
            img_id = file_name.split(".")[0]
            img_path = os.path.join(self.val_data_path, file_name)
            mask_path = os.path.join(self.val_mask_path, img_id + "_mask.png")
            self.examples.append((img_path, mask_path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, mask_path = self.examples[index]
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
