#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import custom_dataset
from torchvision.transforms import v2
import skimage.measure

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")

# select device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define path for real and fake dataset
imgs_real_path = Path("image_active_dataset/imgs_labels_groups.zarr")

# select channel
channel = ["AGP","DNA", "ER"]#, "Mito"]#, "RNA"]
channel.sort()
map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
id_channel = np.array([map_channel[ch] for ch in channel])

dataset_real = custom_dataset.ImageDataset(imgs_real_path,
                            channel,
                            fold_idx=None,
                            img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                 torch.tensor(img, dtype=torch.float32))]),
                                                      # v2.Normalize(mean=len(channel)*[0.5],
                                                      #              std=len(channel)*[0.5])]),
                            label_transform=lambda label: torch.tensor(label, dtype=torch.long))



entropy = skimage.measure.shannon_entropy(img)
