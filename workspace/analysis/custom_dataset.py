import torch
from torch.utils.data import Dataset
import zarr

class ImageDataset(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.channel = channel
        self.fold_idx = fold_idx
        self.img_transform = img_transform
        self.label_transform = label_transform
    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr["imgs"].oindex[self.fold_idx[idx], self.channel]
        labels = self.imgs_zarr["labels"].oindex[self.fold_idx[idx]]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, labels

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
