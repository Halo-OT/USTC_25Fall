"""
数据增强模块
"""
import torch
import random


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return torch.flip(img, dims=[2])
        return img


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return torch.flip(img, dims=[1])
        return img


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
    def __call__(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        elif angle == 90:
            return torch.flip(img.transpose(1, 2), dims=[2])
        elif angle == 180:
            return torch.flip(torch.flip(img, dims=[1]), dims=[2])
        elif angle == 270:
            return torch.flip(img.transpose(1, 2), dims=[1])
        return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def get_train_augmentation():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(angles=[0, 90, 180, 270]),
    ])
