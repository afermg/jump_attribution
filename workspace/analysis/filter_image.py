#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import custom_dataset
from torchvision.transforms import v2
import skimage.measure
import polars as pl

import zarr
from itertools import groupby, starmap

import matplotlib.pyplot as plt
from matplotlib import colors

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")



moa_to_pert = (metadata_pre.select(pl.col(["moa", "moa_id", "pert_iname"]))
               .unique()
               .sort(by=["moa_id","pert_iname"])
               .group_by("moa_id", maintain_order=True)
               .agg(pl.col(["moa", "pert_iname"]).unique(maintain_order=True))
               .with_columns(pl.col("pert_iname").list.sort(),
                             pl.col("moa").list.explode())).to_pandas()


# Plot as a colored table
fig, ax = plt.subplots(figsize=(12, 6))  # Bigger figure size
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(
    cellText=moa_to_pert.values,
    colLabels=moa_to_pert.columns,
    cellLoc='center',
    loc='center',
)

# Apply alternating row colors
for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] > 0:  # Skip the header row
        if key[0] % 2 == 0:  # Even rows
            cell.set_facecolor(colors.to_rgba('white', alpha=1))
        else:  # Odd rows
            cell.set_facecolor(colors.to_rgba('gold', alpha=0.3))
    else:
        cell.set_text_props(weight='bold')
    cell.set_height(0.15)  # Increase cell height
    cell.set_fontsize("large")  # Set font size

# Increase column width to make bigger boxes
table.auto_set_column_width(col=list(range(len(moa_to_pert.columns))))

ax.set_title("Map table from moa_id to perturbation", fontsize="x-large", weight="bold")
fig.savefig(fig_directory / "moa_id_to_pert")
plt.close()


metadata_pre = pl.read_csv("target2_eq_moa2_active_metadata")
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
                                           id_channel,
                                           fold_idx=None,
                                           img_transform=lambda img: np.mean(img, axis=-3), # v2.Compose([v2.Lambda(lambda img:
                                          #                        torch.tensor(img, dtype=torch.float32))]),
                                                      # v2.Normalize(mean=len(channel)*[0.5],
                                                      #              std=len(channel)*[0.5])]),
                                           label_transform=None# lambda label: torch.tensor(label, dtype=torch.long)
                                           )
# import skimage.measure
# entropy = skimage.measure.shannon_entropy(dataset_real[2][0])

# dict1 = {k: g for k,g in list(zip(*list(((metadata_pre
#  .select(pl.col(["moa", "moa_id", "Metadata_InChIKey"]))
#  .unique()
#  .sort(by=["moa_id","Metadata_InChIKey"])
#  .group_by("moa_id", maintain_order=True)
#  .agg(pl.col(["moa", "Metadata_InChIKey"]).unique(maintain_order=True))
#  .select(pl.col("moa_id", "Metadata_InChIKey")))
#  .to_dict(as_series=False)
#  .values()))))}

# imgs_zarr = zarr.open(imgs_real_path)

# dict2 = {k: g for k, g in
#  starmap(lambda k, g: (k, list(map(lambda x: x[1], g))),
#         groupby(sorted(list(set(list(zip(imgs_zarr["labels"][:], imgs_zarr["groups"][:])))),
#                             key=lambda x:(x[0], x[1])),
#                      key=lambda x:(x[0])))}
