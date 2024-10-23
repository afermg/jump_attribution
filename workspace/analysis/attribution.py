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

# select device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define path for real and fake dataset
batch_size = 32
fold = 0
split = "train"
mode="ref"
use_ema = True
fake_img_path_preffix = "image_active_dataset/fake_imgs"

suffix = "_ema" if use_ema else ""
sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"
imgs_fake_path = Path(fake_img_path_preffix + suffix) / sub_directory
imgs_real_path = Path("image_active_dataset/imgs_labels_groups.zarr")

# select channel
channel = ["AGP","DNA", "ER"]#, "Mito"]#, "RNA"]
channel.sort()
map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
id_channel = np.array([map_channel[ch] for ch in channel])

# create dataset_real_fake
dataset_real_fake = custom_dataset.ImageDataset_real_fake(imgs_real_path, imgs_fake_path,
                                                          channel=id_channel,
                                                          org_to_trg_label=None,
                                                          img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                                               torch.tensor(img, dtype=torch.float32)),
                                                                                    v2.Normalize(mean=len(channel)*[0.5],
                                                                                                 std=len(channel)*[0.5])]),
                                                          label_transform=lambda label: torch.tensor(label, dtype=torch.long))
# load trained classifier
VGG_path = "VGG_image_active_fold_0epoch=78-train_acc=0.96-val_acc=0.91.ckpt"
VGG_module = LightningModelV2.load_from_checkpoint(Path("lightning_checkpoint_log") / VGG_path,
                                                   model=conv_model.VGG_ch)
VGG_model = VGG_module.model.eval().to(device)

# need to predict the label of the input and then do attribution technique

imgs_real, imgs_fake, labels_real, labels_fake = dataset_real_fake[:10]
