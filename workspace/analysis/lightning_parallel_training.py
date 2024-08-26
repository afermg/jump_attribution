import lightning as L

import os
import numpy as np
import pandas as pd
from more_itertools import unzip
import torch

import torch.nn as nn
import torch.optim as optim


from torch.nn.functional import cross_entropy


## WORK WITH TORCH METRIC FROM LIGHTNING INSTEAD
from torchmetrics.classification import (
MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix)



class LightningModel(L.LightningModule):
    def __init__(self, model, model_param, lr, weight_decay, max_epoch, n_class):
        super().__init__()
        self.model = model(*model_param)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.n_class = n_class
        self.save_hyperparameters(ignore="model")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)
        #save output
        self.training_step_outputs.append((output, target))
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        if self.trainer.is_global_zero:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, 
                     rank_zero_only=True,
                     sync_dist=True) #do Ihave to use global_zero and sync dist? 
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)
        self.validation_step_outputs.append((output, target))
        if self.trainer.is_global_zero:
            self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, 
                     rank_zero_only=True,
                     sync_dist=True) #do Ihave to use global_zero and sync dist? 


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_train_epoch_end(self):
        self._shared_eval(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self): 
        self._shared_eval(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()  # free memory
        
    def _shared_eval(self, prefix_step_outputs, prefix):
        all_preds, all_labels = map(lambda x: list(x), unzip(prefix_step_outputs))
        all_preds, all_labels = self.all_gather(torch.vstack(all_preds)), self.all_gather(torch.hstack(all_labels))
        all_preds = nn.Softmax(dim=1)(all_preds)
        metrics = {"acc": MulticlassAccuracy(num_classes=self.n_class, average=None), 
                   "roc": MulticlassAUROC(num_classes=self.n_class, average=None), 
                   "f1": MulticlassF1Score(num_classes=self.n_class, average=None)}
        for (token, func) in list(metrics.items()):
            self.log(prefix + "_" + token, func(all_preds, all_labels))
                     
        if self.current_epoch==self.max_epoch:
            self.log(prefix + "_ConfusionMatrix", MulticlassConfusionMatrix(num_classes=self.n_class)(
                all_preds, all_labels))


