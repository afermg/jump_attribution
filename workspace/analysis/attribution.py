#!/usr/bin/env python
# coding: utf-8
import numpy as np

import torch
import torch.nn as nn

from captum.attr import Saliency, IntegratedGradients, DeepLift, GuidedBackprop, GuidedGradCam, LayerGradCam
from captum._utils.gradient import compute_layer_gradients_and_eval
torch.manual_seed(42)
np.random.seed(42)

# General Attribution function
def Attribution(model, method, inputs, baselines, inputs_targe):
    grad_func = method(model)
    return grad_func.attribute(inputs=inputs, baselines=baselines, target=inputs_target)

# D_INGRADS (Input * Gradient)
class D_INGRADS(Saliency):
    def __init__(self, model):
        super().__init__(model)

    def attribute(self, inputs, baselines, target):
        # Call Saliency's attribute method to get the gradients
        ingrad = super().attribute(inputs=inputs, target=target)
        # Modify the gradients as per your original D_INGRADS function
        return torch.abs(ingrad * (inputs - baselines))

# D_IG (Integrated Gradient)

# D_DL (DeepLift)

# D_GC (GradCAM)

# D_GGC (GuidedGradCAM)


# class D_GCC():
#     def __init__(self, model):
#         layer_name, layer = next((name, module) for name, module in reversed(list(model.named_modules())) if isinstance(module, torch.nn.Conv2d))



# D_LGC (LayerGradCAM)

# D_GBP (GuidedBackprop)

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
import torch.nn.functional as F

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

dataloader_real_fake = DataLoader(dataset_real_fake, batch_size=batch_size, num_workers=1, persistent_workers=True)

# load trained classifier
VGG_path = "VGG_image_active_fold_0epoch=78-train_acc=0.96-val_acc=0.91.ckpt"
VGG_module = LightningModelV2.load_from_checkpoint(Path("lightning_checkpoint_log") / VGG_path,
                                                   model=conv_model.VGG_ch)
VGG_model = VGG_module.model.eval().to(device)

# need to predict the label of the input and then do attribution technique
for X_real, X_fake, y_real, y_fake in dataloader_real_fake:
    X_real, X_fake, y_fake = X_real.to(device), X_fake.to(device), y_fake.to(device)
    X_real.requires_grad, X_fake.requires_grad = True, True
    # with torch.no_grad():
    #     y_hat_real = F.softmax(VGG_model(X_real), dim=1).argmax(dim=1)
    #     y_hat_fake = F.softmax(VGG_model(X_fake), dim=1).argmax(dim=1)
    # saliency, grad_fake = D_INGRADS(VGG_model, X_real, X_fake, y_fake)
    break


layer_name, layer = next((name, module) for name, module in reversed(list(VGG_model.named_modules())) if isinstance(module, torch.nn.Conv2d))
VGG_model.zero_grad()
# compute_layer_gradients_and_eval(VGG_model,
#                                  layer,
#                                  X_real,
#                                  y_real,

#                                  )
