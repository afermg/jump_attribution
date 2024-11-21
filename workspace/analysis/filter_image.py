#!/usr/bin/env python3
from collections.abc import Callable
from itertools import groupby, product, starmap
from pathlib import Path
from typing import List

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import polars as pl
import zarr
from jump_portrait.fetch import get_jump_image
from jump_portrait.utils import batch_processing, parallel
from matplotlib import colors
# import skimage.measure
from skimage.util.shape import view_as_windows
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
        # verbose: bool=True,
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
    img_list = parallel(iterable, batch_processing(try_function(get_jump_image)))
                        # verbose=verbose)

    return iterable, img_list

# if not os.path.exists(Path("image_active_dataset/imgs_labels_groups.zarr")):

channel = ['AGP', 'DNA']#, 'ER', 'Mito', 'RNA']
iterable, img_list = get_jump_image_batch(metadata.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                                       "Metadata_Plate", "Metadata_Well"])),
                                          channel=channel,#, 'ER', 'AGP', 'Mito', 'RNA'],
                                          site=["1", "2"],#[str(i) for i in range(1, 7)],
                                          correction=None) #None, 'Illum'
mask = [x is not None for x in img_list]
iterable_mask, img_list_mask = [it for i, it in enumerate(iterable) if mask[i]],  [img for i, img in enumerate(img_list) if mask[i]]

zip_iter_img = sorted(zip(iterable_mask, img_list_mask),
                      key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][5], x[0][4]))
# grouped image are returned as the common key, and then the zip of param and img, so we retrieve the img then we stack
iterable_stack, img_list_stack = map(lambda tup: list(tup),
                                zip(*starmap(lambda key, param_img: (key, np.stack(list(map(lambda x: x[1], param_img)))),
                                             groupby(zip_iter_img,
                                                     key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][5])))))

# retrieve moa_id and InChIKey
labels, groups = tuple((metadata
        .select(pl.col(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "moa_id", "Metadata_InChIKey"]))
        .join(pl.DataFrame([t[:4] for t in iterable_stack],
                           schema=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]),
              on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
              how="left")
        .select(pl.col(["moa_id", "Metadata_InChIKey"]))
        .to_dict(as_series=False)
        .values()))

# crop img in small square
list_shape = np.array(list(map(lambda x: min(*x.shape[-2:]), img_list_stack)))
wanted_crop = 128
num_crop = 8 #int(np.round(np.median(list_shape/ wanted_crop)))

def window_step(img, window_length, window_num):
    length = min(*img.shape[-2:])
    return (length - window_length) // (window_num -1)

img_stack_window = np.vstack(list(map(
    lambda img: view_as_windows(img,
                                (img.shape[-3], wanted_crop, wanted_crop),
                                window_step(img, wanted_crop, num_crop)),
    img_list_stack)))

pixel_window =  sum(list(map(np.size,  img_stack_window)))
pixel_origin = sum(list(map(np.size,  img_list_stack)))
print(f"pixel increase due to overlapping window: {(pixel_window - pixel_origin) / pixel_origin:.2%}")

# clip image
# axis of img_stack_window = (sample, row_grid, column_grid, channel, H, W)
clip = (1, 99)
min_clip_img, max_clip_img = np.percentile(img_stack_window, clip, axis=(1, 2, 4, 5), keepdims=True)
img_stack_window_norm = np.clip(img_stack_window,
                                min_clip_img,
                                max_clip_img)
img_stack_window_norm = (img_stack_window_norm - min_clip_img) / (max_clip_img - min_clip_img)


img_flat_crop = img_stack_window_norm.reshape(-1, *img_stack_window_norm.shape[-3:])
labels_flat_crop = np.repeat(labels, num_crop ** 2)
groups_flat_crop = np.repeat(groups, num_crop ** 2)

def channel_to_rgb(img_array: np.ndarray,
                   channel: List[str]):
    if img_array.ndim not in {3, 4}:
        raise Exception(
            f"input array should have shape (sample, C, H, W) or (C, H, W).\n"
            f"Here input shape: {img_array.shape}")
    if img_array.shape[-3] > len(channel):
        raise Exception(f"input array should have shape (sample, C, H, W) or (C, H, W) with C <= 5.\n"
                        f"Here input shape: {img_array.shape}\n"
                        f"And C: {img_array.shape[-3]}")

    map_channel_color = {
        "AGP": "#FFA500",
        "DNA": "#0000FF",
        "ER": "#00FF00", #"#65fe08"
        "Mito": "#FF0000",
        "RNA": "#FFFF00",
                         }
    channel_rgb_weight = np.vstack(list(map(lambda ch: list(mcolors.to_rgb(map_channel_color[ch])), channel)))
    img_array_rgb = np.tensordot(img_array, channel_rgb_weight, axes=[[-3], [0]])
    if len(img_array_rgb.shape) == 4:
        img_array_rgb = img_array_rgb.transpose(0, 3, 1, 2)
    else:
        img_array_rgb = img_array_rgb.transpose(2, 0, 1)
    return img_array_rgb


# filter out low quality image
def plot_img(img_array, label_array, channel, size=4, fig_name="multiple_cells_small_crop", fig_directory=Path("./figures")):
    rng = np.random.default_rng(seed=42)
    choice = rng.choice(img_array.shape[0], size=size*size, replace=False)
    fig, axis = plt.subplots(size, size, figsize=(5 * size, 5 * size))
    axis = axis.flatten()
    for ax, i in zip(axis, choice):
        ax.imshow(channel_to_rgb(img_array[i], channel).transpose(1, 2 ,0))
        ax.set_axis_off()
        ax.set_title(f"class: {label_array[i]}")
    fig.savefig(fig_directory / fig_name)
    plt.close()

plot_img(img_flat_crop, labels_flat_crop, channel=channel, size=4, fig_name="multiple_cells_small_crop", fig_directory=fig_directory)

# store = zarr.DirectoryStore(Path("image_active_dataset/imgs_labels_groups.zarr"))
# root = zarr.group(store=store)
# # Save datasets into the group
# root.create_dataset('imgs', data=small_image, overwrite=True, chunks=(1, 1, *small_image.shape[2:]))
# root.create_dataset('labels', data=small_labels, overwrite=True, chunks=1)
# root.create_dataset('groups', data=small_groups, dtype=object, object_codec=numcodecs.JSON(), overwrite=True, chunks=1)

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
