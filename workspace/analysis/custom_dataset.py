import numpy as np
import torch
from torch.utils.data import Dataset
import zarr
from itertools import groupby

class ImageDataset(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
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

class ImageDataset_all_info(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
        self.channel = channel
        self.fold_idx = fold_idx
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr["imgs"].oindex[self.fold_idx[idx], self.channel]
        labels = self.imgs_zarr["labels"].oindex[self.fold_idx[idx]]
        groups = self.imgs_zarr["groups"].oindex[self.fold_idx[idx]]
        indices = self.fold_idx[idx]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, labels, groups, indices

class ImageDataset_Ref(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None, seed=42):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
        self.channel = channel
        self.fold_idx = fold_idx
        self.imgs_idx, self.imgs2_idx = self._make_dataset(seed)
        self.img_transform = img_transform
        self.label_transform = label_transform

    def _make_dataset(self, seed):
        labels = self.imgs_zarr["labels"].oindex[self.fold_idx]
        domain_idx = sorted(zip(self.fold_idx, labels), key=lambda x: x[1])
        domain_idx = {k: np.array(list(zip(*g))[0]) for k, g in groupby(domain_idx,  key=lambda x:x[1])}
        rng = np.random.default_rng(42)
        return list(map(lambda list_idx: np.concatenate(list_idx),
                        list(zip(*[(group_idx,
                                    rng.choice(group_idx, size=len(group_idx), replace=False))
                                   for group_idx in list(domain_idx.values())]))))

    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr["imgs"].oindex[self.imgs_idx[idx], self.channel]
        imgs2 = self.imgs_zarr["imgs"].oindex[self.imgs2_idx[idx], self.channel]
        labels = self.imgs_zarr["labels"].oindex[self.imgs_idx[idx]]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
            imgs2 = self.img_transform(imgs2)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, imgs2, labels


class ImageDataset_fake(Dataset):
    def __init__(self, imgs_path, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.length = self.imgs_zarr["imgs"].shape[0]
        self.num_outs_per_domain = self.imgs_zarr["imgs"].shape[1]
        self.imgs_path = imgs_path
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return self.length * self.num_outs_per_domain

    def __getitem__(self, idx):
        indices = np.arange(self.length * self.num_outs_per_domain)[idx]
        true_idx, img_rank = np.divmod(indices, self.num_outs_per_domain)
        if len(indices.shape) == 0 :
            imgs = self.imgs_zarr["imgs"].oindex[true_idx, img_rank]
        else:
            imgs = np.stack(list(map(lambda x: imgs_zarr["imgs"][*x], zip(true_idx, img_rank))))
        labels = self.imgs_zarr["labels"].oindex[true_idx]
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
