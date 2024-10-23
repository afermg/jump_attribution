#!/usr/bin/env python
# coding: utf-8
import numpy as np

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, Saliency

torch.manual_seed(123)
np.random.seed(123)

# D_INGRADS (Input * Gradient)

def D_INGRADS(model, X_real, X_fake, y_fake):
    grads_func = Saliency(model)
    grads_fake = grad_func.attribute(input=X_fake, target=y_fake)
    return torch.abs(grads_fake * (X_real - X_fake))


# D_IG (Integrated Gradient)
# D_DL (DeepLift)
# D_GC (GradCAM)
# D_GGC (GuidedGradCAM)
# Random
# Residual
"""
-----------------------------------------------------------
"""


import zarr

from itertools import starmap
from data_split import StratifiedGroupKFold_custom
from sklearn.model_selection import StratifiedShuffleSplit

from torchvision.transforms import v2
from torch.utils.data import DataLoader, BatchSampler
import conv_model
import custom_dataset
from lightning_parallel_training import LightningModelV2

from pathlib import Path


# Test Method

image_dataset = zarr.open(Path("image_active_dataset/imgs_labels_groups.zarr"))
kfold_train_test = list(StratifiedGroupKFold_custom(random_state=42).split(None, image_dataset["labels"][:], image_dataset["groups"][:]))
kfold_train_val_test = list(starmap(lambda train, val_test: (train,
                                                             *list(starmap(lambda val, test: (val_test[val], val_test[test]),
                                                                           StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.5).split(
                                                                               np.zeros(len(val_test)), image_dataset["labels"].oindex[val_test])))[0]),
                                    kfold_train_test))
# ## #i) Transformation applied to train split
img_transform_train = v2.RandomApply([v2.RandomVerticalFlip(p=0.5),
                                      v2.RandomChoice([v2.Lambda(lambda img: v2.functional.rotate(img, angle=0)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=90)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=180)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=270))])],
                                     p=1)

fold_L = np.arange(5)
channel = ["AGP","DNA", "ER"]#, "Mito"]#, "RNA"]
channel.sort()
map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
id_channel = np.array([map_channel[ch] for ch in channel])
imgs_path = Path("image_active_dataset/imgs_labels_groups.zarr")

def create_dataset_fold(Dataset, imgs_path, id_channel, kfold_train_val_test, img_transform_train):
    return {i: {"train": Dataset(imgs_path,
                                 channel=id_channel,
                                 fold_idx=kfold_train_val_test[i][0],
                                 img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                           img_transform_train,
                                                           v2.Normalize(mean=len(channel)*[0.5],
                                                                                std=len(channel)*[0.5])]),
                                 label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                "val": Dataset(imgs_path,
                               channel=id_channel,
                               fold_idx=kfold_train_val_test[i][1],
                               img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                        v2.Normalize(mean=len(channel)*[0.5],
                                                                     std=len(channel)*[0.5])]),
                               label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                "test": Dataset(imgs_path,
                                channel=id_channel,
                                fold_idx=kfold_train_val_test[i][2],
                                img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                          v2.Normalize(mean=len(channel)*[0.5],
                                                                       std=len(channel)*[0.5])]),
                                label_transform=lambda label: torch.tensor(label, dtype=torch.long))}
            for i in fold_L}


dataset_fold = create_dataset_fold(custom_dataset.ImageDataset, imgs_path, id_channel, kfold_train_val_test,
                                   img_transform_train)


VGG_path = "VGG_image_active_fold_0epoch=78-train_acc=0.96-val_acc=0.91.ckpt"
VGG_module = LightningModelV2.load_from_checkpoint(Path("lightning_checkpoint_log") / VGG_path,
                                                   model=conv_model.VGG_ch)

batch_size = 32
fold = 0
split = "train"

use_ema = True
fake_img_path_preffix = "image_active_dataset/fake_imgs"
mode="ref"
suffix = "_ema" if use_ema else ""
fake_img_path = Path(fake_img_path_preffix + suffix)
sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"
imgs_fake_path = fake_img_path / sub_directory
dataset_fake = custom_dataset.ImageDataset_fake(imgs_fake_path,
                                                img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                                     torch.tensor(img, dtype=torch.float32)),
                                                                          v2.Normalize(mean=len(channel)*[0.5],
                                                                                       std=len(channel)*[0.5])]),
                                                label_transform=lambda label: torch.tensor(label, dtype=torch.long))
imgs_zarr_fake = zarr.open(imgs_fake_path)

org_to_trg_label = [(0, 1), (0, 2), (0, 3), (1, 0)]
mask = np.sum([(imgs_zarr_fake["labels"].oindex[:] == label) &
               (imgs_zarr_fake["labels_org"].oindex[:] == label_org)
               for (label_org, label) in org_to_trg_label], axis=0)
