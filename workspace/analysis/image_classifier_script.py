#!/usr/bin/env python
# coding: utf-8
import os
import polars as pl
import pandas as pd

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from itertools import groupby
from more_itertools import unzip

from data_split import StratifiedGroupKFold_custom

from jump_portrait.fetch import get_jump_image
from jump_portrait.utils import batch_processing, parallel

from collections.abc import Callable, Iterable
from typing import List

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L


from parallel_training import run_train
import conv_model
import custom_dataset
from lightning_parallel_training import LightningModel
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path

# # 1) Loading images and create a pytorch dataset

# ## a) Load Images using Jump_portrait

# In[2]:

file_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")
metadata_pre = pl.read_csv("target2_eq_moa2_metadata")


def try_function(f: Callable):
    '''
    Wrap a function into an instance which will Try to call the function:
        If it success, return a tuple of function parameters + its results
        If it fails, return the function parameters
    '''
    # This assume parameters are packed in a tuple
    def batched_fn(*item, **kwargs):
        try:
            result = (*item, f(*item, **kwargs))
            
        except:
            result = item

        return result    
    return batched_fn


# ### ii) function to overcome turn get_jump_image_iter compatible with list and load data in a threaded fashion

def get_jump_image_iter(metadata: pl.DataFrame, channel: List[str],
                        site:List[str], correction:str=None) -> (pl.DataFrame, List[tuple]): 
    '''
       Load jump image associated to metadata in a threaded fashion.
        ----------
    Parameters: 
        metadata(pl.DataFrame): must have the shape (Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
        channel(List[str]): list of channel desired
            Must be in ['DNA', 'ER', 'AGP', 'Mito', 'RNA']
        site(List[str]): list of site desired
            For compound, must be in ['1' - '6']
            For ORF, CRISPR, must be in ['1' - '9']
        correction(str): Must be 'Illum' or None
        ----------
    Return:
        features(pl.DataFrame): DataFrame collecting the metadata, channel, site, correction + the image
        work_fail(List(tuple): List collecting tuple of metadata which failed to load an image
        
    '''
    iterable = [(*metadata.row(i), ch, s, correction)
               for i in range(metadata.shape[0]) for s in site for ch in channel]
    img_list = parallel(iterable, batch_processing(try_function(get_jump_image)))
    
    img_list = sorted(img_list, key=lambda x: len(x))
    fail_success = {k: list(g) for k, g in groupby(img_list, key=lambda x: len(x))}
    if len(fail_success) == 1:
        img_success = list(fail_success.values())[0]
        work_fail = []
    else: 
        work_fail, img_success = fail_success.values()
    features = pl.DataFrame(img_success, 
                               schema=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
                                        "channel", "site", "correction",
                                        "img"])
    return features, work_fail


features_pre, work_fail = get_jump_image_iter(metadata_pre.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                                               "Metadata_Plate", "Metadata_Well"])),
                                                        channel=['DNA'],#, 'ER', 'AGP', 'Mito', 'RNA'],
                                                        site=[str(i) for i in range(1, 7)],
                                                        correction=None) #None, 'Illum'


# ### iii) Add 'site' 'channel' and filter out sample which could not be load (using join)

metadata = metadata_pre.join(features_pre.select(pl.all().exclude(["correction", "img"])),
              on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
              how="inner").sort(by=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site"])

metadata = metadata.with_columns(pl.arange(0,len(metadata)).alias("ID"))
features = features_pre.join(metadata.select(pl.col(["Metadata_Source", "Metadata_Batch", 
                                                 "Metadata_Plate", "Metadata_Well", "site", "ID"])),
                         on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", 
                             "Metadata_Well", "site"],
                         how="inner").sort(by="ID")


# ## b) Visualisation of some images

fig, axes = plt.subplots(5, 4, figsize=(30,30))
axes = axes.flatten()
select = np.random.randint(0, len(features), (20))
for i in range(20):
    axes[i].imshow(features["img"][int(select[i])])
fig.savefig(file_directory / "multiple_cells", dpi=300)
plt.close()

#Visualisation of the chosen kernel size relative to the image size
fig, axes = plt.subplots(1, 1, figsize=(60,60))
axes.imshow(features["img"][1])
rect = patches.Rectangle((335, 320), 46, 46, linewidth=7, edgecolor='r', facecolor='none')
axes.add_patch(rect)
fig.savefig(file_directory / "kernel_vs_image_size", dpi=300)
plt.close()

# ## c) Resize images

def crop_square_array(x, des_size):
    h, w = x.shape
    h_off, w_off = (h-des_size)//2, (w-des_size)//2
    return x[h_off:des_size+h_off,w_off:des_size+w_off]


shape_image = list(map(lambda x: x.shape, features["img"].to_list()))
shape_image.sort(key=lambda x:x[0])
shape_image_count = {k: shape_image.count(k) for k in set(shape_image)}
image_resized = np.array(list(map(lambda x: crop_square_array(x, shape_image[0][0]), features["img"].to_list())))


# ## d) Encode moa intp moa_id
metadata_df = metadata.to_pandas().set_index(keys="ID")
metadata_df = metadata_df.assign(moa_id=LabelEncoder().fit_transform(metadata_df["moa"]))
labels, groups = metadata_df["moa_id"].values, metadata_df["Metadata_InChIKey"].values


# ## e) Split image in 4

image_size = image_resized.shape[1]
small_image_resized = np.vstack([image_resized[:, :image_size//2, :image_size//2], 
           image_resized[:, :image_size//2, image_size//2:],
           image_resized[:, image_size//2:, :image_size//2],
           image_resized[:, image_size//2:, image_size//2:]])
small_labels = np.hstack([labels for i in range(4)])
small_groups = np.hstack([groups for i in range(4)])


# ## f) Create the pytorch dataset with respect to kfold split


kfold = list(StratifiedGroupKFold_custom().split(small_image_resized, small_labels, small_groups))


small_image_resized_tensor = torch.unsqueeze(torch.tensor(small_image_resized, dtype=torch.float), 1)
small_labels_tensor = torch.tensor(small_labels, dtype=torch.long)
dataset_fold = {i: {"train": custom_dataset.ImageDataset(small_image_resized_tensor[kfold[i][0]], 
                                                         small_labels_tensor[kfold[i][0]]),
     "test": custom_dataset.ImageDataset(small_image_resized_tensor[kfold[i][1]], small_labels_tensor[kfold[i][1]])}
 for i in range(len(kfold))}


# ### i) Memory usage per fold

def fold_memory_usage(fold:int, split:str ="train", batch_size:int=None):
    total_size = 0
    for i in range(len(dataset_fold[fold][split])):
        sample = dataset_fold[fold][split][i]
        sample_size = 0
        if isinstance(sample, tuple):
            for item in sample:
                if isinstance(item, torch.Tensor):
                    sample_size += item.element_size() * item.nelement()
        total_size += sample_size
    if batch_size is not None:
        normalizer = batch_size / len(dataset_fold[fold][split])
    else:
        normalizer = 1
    print(f"Total fold {fold} size for {split} set with {batch_size} "+
          f"batch_size: {normalizer * total_size / (1024 ** 2):.2f} MB")
    
fold_memory_usage(0, "train", None)


# # 2) Model

# ## a) Receptive field calculation
# Receptive field are important to visualise what information of the original image is convolve to get end features 
# Computation of the receptive field is base don this [article](https://distill.pub/2019/computing-receptive-fields/). 

def compute_receptive_field(model):
    dict_module = dict(torch.fx.symbolic_trace(model).named_modules())
    def extract_kernel_stride(module):
        try:
            return (module.kernel_size[0] if type(module.kernel_size) == tuple else module.kernel_size, 
                    module.stride[0] if type(module.stride) == tuple else module.stride) 
        except:
            return None
    
    k, s = list(map(lambda x: np.array(list(x)), 
                    unzip([x for x in list(map(extract_kernel_stride, list(dict_module.values()))) if x is not None])))
    return ((k-1) * np.concatenate((np.array([1]), s.cumprod()[:-1]))).sum() + 1

def compute_receptive_field_recursive(model):
    dict_module = dict(torch.fx.symbolic_trace(model).named_modules())
    def extract_kernel_stride(module):
        try:
            return (module.kernel_size[0] if type(module.kernel_size) == tuple else module.kernel_size, 
                    module.stride[0] if type(module.stride) == tuple else module.stride) 
        except:
            return None
    
    k, s = list(map(lambda x: np.array(list(x))[::-1], 
                    unzip([x for x in list(map(extract_kernel_stride, list(dict_module.values()))) if x is not None])))

    res = [1, k[0]]
    for i in range(1, len(k)):
        res.append(s[i]*res[i]+k[i]-s[i])
    return res

# ## b) Memory usage calculation
# Memory is the bottleneck in GPU training so knowing the size of the model is important

def model_memory_usage(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.2f}MB'.format(size_all_mb))
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"Free memory in cuda 0 before model load: {free_mem / 1024**2:.2f} MB")
    free_mem = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_allocated(1)
    print(f"Free memory in cuda 1 before model load: {free_mem / 1024**2:.2f} MB")


vgg = conv_model.VGG(img_depth=1,
          img_size=485, 
          lab_dim=7, 
          n_conv_block=4,
          n_conv_list=[1 for _ in range(4)],
          n_lin_block=3)

vgg_ch = conv_model.VGG_ch(
          img_depth=1,
          img_size=485,
          lab_dim=7,
          conv_n_ch=64,
          n_conv_block=6,
          n_conv_list=[2, 2, 2, 2, 4, 4],
          n_lin_block=3,
          p_dropout=0.2)

print(f'recursive receptive field from very end to start: {compute_receptive_field_recursive(vgg_ch)}')
recept_field = compute_receptive_field(vgg_ch)
print(f'recursive receptive field at the start: {recept_field}')
#model_memory_usage(vgg)



#Visualisation of the chosen kernel size relative to the image size
fig, axes = plt.subplots(1, 1, figsize=(30,30))
axes.imshow(dataset_fold[0]["train"][1][0][0])
rect = patches.Rectangle((130, 280), recept_field, recept_field, linewidth=7, edgecolor='r', facecolor='none')
axes.add_patch(rect)
fig.savefig(file_directory / "kernel_vgg_vs_small_image_size")
plt.close()

# ### No need for parallel model
# ### Let's however speed up training by taking advantage of the 2 GPU. 

# # 3) Trainer

# ## a) DistributedDataParallel

# run_train(model=conv_model.VGG,
#           model_param=(1, #img_depth
#                        970, #img_size
#                        7, #lab_dim
#                        6, #n_conv_block
#                        [2, 2, 2, 2, 2, 2], #n_conv_list
#                        3), #n_lin_block
#           adam_param={"lr": 1e-3,
#                       "weight_decay":0}, 
#           dataset=dataset_fold[0]["train"],
#           batch_size=16, 
#           epochs=100, 
#           fold=0,
#           filename='ddp_trained_model_fold_',
#           log_loss={})


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# torch.cuda.empty_cache()
# res_df = run_train(
#           model=conv_model.VGG,
#           model_param=(1, #img_depth
#                        485, #img_size
#                        7, #lab_dim
#                        5, #n_conv_block
#                        [1, 1, 1, 1, 1], #n_conv_list
#                        3), #n_lin_block
#           adam_param={"lr": 1e-3,
#                       "weight_decay":0},
#           train_dataset=dataset_fold[0]["train"],
#           test_dataset=dataset_fold[0]["test"],
#           batch_size=512,
#           epochs=3,
#           fold=0,
#           filename='vgg8',
#           allow_eval=True
# )


# res_df = pd.read_pickle("trained_model/vgg8_fold_0_result.pkl")


# def train_test_line_plot(train_arr, test_arr, ax, title, ylabel):
#     num_epoch = len(train_arr)
#     sns.lineplot(train_arr, ax=ax, label="train")
#     sns.lineplot(test_arr, ax=ax, label="test")
#     ax.legend(title="split")
#     ax.set_title(title)
#     ax.set_xlabel("epochs")
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(np.arange(num_epoch), np.arange(num_epoch))
#     return ax


# fig, axes = plt.subplots(3,3, figsize=(20, 19))
# axes[0][0] = train_test_line_plot(res_df.loc["losses"]["train"], res_df.loc["losses"]["test"],
#                      ax=axes[0][0],
#                      title="losses across epochs",
#                      ylabel="losses")
# for i, split in enumerate(["train", "test"]):
#     sns.heatmap(res_df.loc["matrix"][split], ax=axes[0][i+1], annot=True, fmt=".0f")
#     axes[0][i+1].set_xlabel("predicted label")
#     axes[0][i+1].set_ylabel("true label")
#     axes[0][i+1].set_title(f"{split} confusion matrix last epoch")
# for i, metric_key in enumerate(["acc", "roc", "f1"]):
#     metric_train, metric_test = res_df.loc[metric_key]["train"], res_df.loc[metric_key]["test"]
#     axes[1][i] = train_test_line_plot(metric_train.mean(axis=1), metric_test.mean(axis=1),
#                                       ax=axes[1][i],
#                                       title=f"avg {metric_key} across epoch",
#                                       ylabel=metric_key)

#     # Combine train and test last epoch data into a DataFrame
#     num_class = metric_train.shape[1]
#     df = pd.DataFrame({
#         'class': list(range(num_class)) + list(range(num_class)),
#         metric_key: list(metric_train[-1, :]) + list(metric_test[-1, :]),
#         'split': ["train"] * num_class + ["test"] * num_class
#     })

#     # Create the barplot
#     sns.barplot(x='class', y=metric_key, hue='split', data=df, ax=axes[2][i])
#     axes[2][i].set_title(f"{metric_key} last epoch across classes")
# # Display the plot
# plt.show()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # Lightning Training
tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"))
checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
                                      every_n_epochs=10)
#To try
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

torch.set_float32_matmul_precision('medium') #try 'high')
seed_everything(42, workers=True)
# lit_model = LightningModel(conv_model.VGG,
#                            model_param=(1, #img_depth
#                                         485, #img_size
#                                         7, #lab_dim
#                                         5, #n_conv_block
#                                         [1, 1, 1, 1, 1], #n_conv_list
#                                         3),
#                            lr=5e-4,
#                            weight_decay=5e-3,
#                            max_epoch=1,
#                            n_class=7)

lit_model = LightningModel(conv_model.VGG_ch,
                           model_param=(1, #img_depth
                                        485, #img_size
                                        7, #lab_dim
                                        32, #conv_n_ch
                                        6, #n_conv_block
                                        [1, 1, 2, 2, 3, 3], #n_conv_list
                                        3, #n_lin_block
                                        0.2), #p_dropout
                           lr=5e-4,
                           weight_decay=0,
                           max_epoch=1,
                           n_class=7)

trainer = L.Trainer(#default_root_dir="./lightning_checkpoint_log/",
                    accelerator="gpu",
                    devices=2,
                    strategy="ddp_notebook",
                    max_epochs=50,
                    logger=tb_logger,
                    #profiler="simple",
                    num_sanity_val_steps=0, #to use only if you know the trainer is working !
                    callbacks=[checkpoint_callback],
                    #enable_checkpointing=False,
                    enable_progress_bar=False
                    )

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

trainer.fit(lit_model, DataLoader(dataset_fold[0]["train"], batch_size=128, num_workers=1, shuffle=True, persistent_workers=True),
            DataLoader(dataset_fold[0]["test"], batch_size=128, num_workers=1, persistent_workers=True))
