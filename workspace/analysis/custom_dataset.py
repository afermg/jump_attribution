import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None, target_transform=None):
        super().__init__()
        self.imgs = imgs
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx): 
        return self.imgs[idx,:,:], self.img_labels[idx]


class RowDataset(Dataset):
    def __init__(self, row, labels, transform=None, target_transform=None):
        super().__init__()
        self.row = row
        self.row_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.row_labels)

    def __getitem__(self, idx): 
        return self.row[idx,:], self.row_labels[idx]