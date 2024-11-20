#!/usr/bin/env python3
import itertools
from collections.abc import Callable, Iterable
from itertools import groupby, product, starmap
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import polars as pl
import seaborn as sns
import skimage.measure
import torch
import zarr
from jump_portrait.fetch import get_jump_image
from jump_portrait.utils import batch_processing, parallel
from matplotlib import colors
from more_itertools import unzip
from torchvision.transforms import v2

import custom_dataset

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")


"""
 --- plot the mapping between moa_id to perturbation into a figure to keep this mapping in mind ---
"""
def plot_table_id_to_pert(metadata: pl.DataFrame,
                          fig_name: str="moa_id_to_pert",
                          fig_directory: str=Path("./figures")):
    moa_to_pert = (metadata.select(pl.col(["moa", "moa_id", "pert_iname"]))
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

    fig.savefig(fig_directory / fig_name)
    plt.close()

metadata = pl.read_csv("target2_eq_moa2_active_metadata")
# plot_table_id_to_pert(metadata, fig_name="moa_id_to_pert", fig_directory=fig_directory)

"""
--- Start fetching images and process them ---
"""


def try_function(f: Callable):
    '''
    Wrap a function into an instance which will Try to call the function:
        If it success, return the output of the function.
        If it fails, return None

    Parameters
    ----------
    f : Callable

    Returns
    -------
    tryed_fn : Callable
    '''
    # This assumes parameters are packed in a tuple
    def tryed_fn(*item, **kwargs):
        try:
            result = f(*item, **kwargs)

        except:
            result = None

        return result
    return tryed_fn


# ### ii) function to overcome turn get_jump_image_iter compatible with list and load data in a threaded fashion

def get_jump_image_batch(
        metadata: pl.DataFrame,
        channel: list[str],
        site: list[str],
        correction: str='Orig',
        verbose: bool=True,
) -> tuple[list[tuple], list[np.ndarray]]:
    '''
    Load jump image associated to metadata in a threaded fashion.

    Parameters:
    ----------
    metadata : pl.DataFrame
        must have the column in this specific order ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
    channel : list of string
        list of channel desired
        Must be in ['DNA', 'ER', 'AGP', 'Mito', 'RNA']
    site : list of string
        list of site desired
        - For compound, must be in ['1' - '6']
        - For ORF, CRISPR, must be in ['1' - '9']
    correction : str
        Must be 'Illum' or 'Orig'
    verbose : bool
        Whether to enable tqdm or not.

    Return:
    ----------
    iterable : list of tuple
        list containing the metadata, channel, site and correction
    img_list : list of array
        list containing the images

    '''
    iterable = list(starmap(lambda *x: (*x[0], *x[1:]), product(metadata.rows(), channel, site, [correction])))
    img_list = parallel(iterable, batch_processing(try_function(get_jump_image)),
                        verbose=verbose)

    return iterable, img_list

if not os.path.exists(Path("image_active_dataset/imgs_labels_groups.zarr")):
    channel = ['AGP', 'DNA', 'ER', 'Mito', 'RNA']
    features_pre, work_fail = get_jump_image_iter(metadata_pre.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                                              "Metadata_Plate", "Metadata_Well"])),
                                                  channel=channel,#, 'ER', 'AGP', 'Mito', 'RNA'],
                                                  site=[str(i) for i in range(1, 7)],
                                                  correction=None) #None, 'Illum'


    # ### iii) Add 'site' 'channel' and filter out sample which could not be load (using join)

    correct_index = (features_pre.select(pl.all().exclude("correction", "img"))
                     .sort(by=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site", "channel"])
                     .group_by(by=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site"], maintain_order=True)
                     .all()
                     .with_columns(pl.arange(0,len(features_pre)//len(channel)).alias("ID")))

    metadata = metadata_pre.select(pl.all().exclude("ID")).join(correct_index,
                                                                on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
                                                                how="inner").select("ID", pl.all().exclude("ID")).sort("ID")

    features = features_pre.join(metadata.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                         "Metadata_Plate", "Metadata_Well", "site",  "ID"])),
                                 on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate",
                                     "Metadata_Well", "site"],
                                 how="inner").sort(by=["ID", "channel"])

    # ## #a) crop image and stack them
    def crop_square_array(x, des_size):
        h, w = x.shape
        h_off, w_off = (h-des_size)//2, (w-des_size)//2
        return x[h_off:des_size+h_off,w_off:des_size+w_off]


    # shape_image = list(map(lambda x: x.shape, features["img"].to_list()))
    # shape_image.sort(key=lambda x:x[0])
    img_crop = list(map(lambda x: crop_square_array(x, des_size=896), features["img"].to_list())) #shape_image[0][0]
    img_stack = np.array([np.stack([item[1] for item in tostack])
                 for idx, tostack in groupby(zip(features["ID"].to_list(), img_crop), key=lambda x: x[0])])

    # ## b) Encode moa into moa_id
    metadata_df = metadata.to_pandas().set_index(keys="ID")
    metadata_df = metadata_df.assign(moa_id=LabelEncoder().fit_transform(metadata_df["moa"]))
    labels, groups = metadata_df["moa_id"].values, metadata_df["Metadata_InChIKey"].values

    # ## c) Clip image
    clip = (1, 99)
    min_clip_img, max_clip_img = np.percentile(img_stack, clip, axis=(2,3), keepdims=True)
    clip_image = np.clip(img_stack,
                         min_clip_img,
                         max_clip_img)
    clip_image = (clip_image - min_clip_img) / (max_clip_img - min_clip_img)

    # ## d) Split image in 4
    def slice_image(img, labels, groups):
        image_size = img.shape[-1]
        small_image = np.vstack([img[:, :, :image_size//2, :image_size//2],
                                 img[:, :, :image_size//2, image_size//2:],
                                 img[:, :, image_size//2:, :image_size//2],
                                 img[:, :, image_size//2:, image_size//2:]])
        small_labels = np.hstack([labels for i in range(4)])
        small_groups = np.hstack([groups for i in range(4)])
        return small_image, small_labels, small_groups
    small_image, small_labels, small_groups = slice_image(clip_image, labels, groups)

    store = zarr.DirectoryStore(Path("image_active_dataset/imgs_labels_groups.zarr"))
    root = zarr.group(store=store)
    # Save datasets into the group
    root.create_dataset('imgs', data=small_image, overwrite=True, chunks=(1, 1, *small_image.shape[2:]))
    root.create_dataset('labels', data=small_labels, overwrite=True, chunks=1)
    root.create_dataset('groups', data=small_groups, dtype=object, object_codec=numcodecs.JSON(), overwrite=True, chunks=1)

# # select device
# device = ("cuda" if torch.cuda.is_available() else "cpu")

# # define path for real and fake dataset
# imgs_real_path = Path("image_active_dataset/imgs_labels_groups.zarr")

# # select channel
# channel = ["AGP","DNA", "ER"]#, "Mito"]#, "RNA"]
# channel.sort()
# map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
# id_channel = np.array([map_channel[ch] for ch in channel])


# dataset_real = custom_dataset.ImageDataset(imgs_real_path,
#                                            id_channel,
#                                            fold_idx=None,
#                                            img_transform=lambda img: np.mean(img, axis=-3), # v2.Compose([v2.Lambda(lambda img:
#                                           #                        torch.tensor(img, dtype=torch.float32))]),
#                                                       # v2.Normalize(mean=len(channel)*[0.5],
#                                                       #              std=len(channel)*[0.5])]),
#                                            label_transform=None# lambda label: torch.tensor(label, dtype=torch.long)
#                                            )
# # import skimage.measure
# # entropy = skimage.measure.shannon_entropy(dataset_real[2][0])

# # dict1 = {k: g for k,g in list(zip(*list(((metadata
# #  .select(pl.col(["moa", "moa_id", "Metadata_InChIKey"]))
# #  .unique()
# #  .sort(by=["moa_id","Metadata_InChIKey"])
# #  .group_by("moa_id", maintain_order=True)
# #  .agg(pl.col(["moa", "Metadata_InChIKey"]).unique(maintain_order=True))
# #  .select(pl.col("moa_id", "Metadata_InChIKey")))
# #  .to_dict(as_series=False)
# #  .values()))))}

# # imgs_zarr = zarr.open(imgs_real_path)

# # dict2 = {k: g for k, g in
# #  starmap(lambda k, g: (k, list(map(lambda x: x[1], g))),
# #         groupby(sorted(list(set(list(zip(imgs_zarr["labels"][:], imgs_zarr["groups"][:])))),
# #                             key=lambda x:(x[0], x[1])),
# #                      key=lambda x:(x[0])))}
