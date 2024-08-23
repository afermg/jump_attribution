import lightning as L

import os
import numpy as np
import pandas as pd
from more_itertools import unzip
import torch

import torch.nn as nn
import torch.optim as optim


from torch.nn.functional import cross_entropy

from torcheval.metrics.functional import (
multiclass_accuracy, multiclass_auroc, multiclass_f1_score, multiclass_confusion_matrix)
from tqdm import tqdm




class LightningModel(L.LightningModule):
    def __init__(self, model, lr, weight_decay, max_epoch):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)
        #save output
        self.training_step_outputs.append(output)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)
        self.test_step_outputs.append((output, target))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, self.weight_decay)
        return optimizer

    def on_train_epoch_end(self):
        all_preds, all_labels = list(unzip(self.training_step_outputs))
        all_preds = nn.Softmax(dim=1)(all_preds)
        metrics = {"acc": multiclass_accuracy, 
                   "roc": multiclass_auroc, 
                   "f1": multiclass_f1_score}
        for (token, func) in list(metrics.items()):
            self.log("train_" + token, func(all_preds, all_labels, 
                                            num_classes=all_preds.shape[1], 
                                            average=None))
        if self.current_epoch==self.max_epoch:
            (mode, "matrix", 
                             multiclass_confusion_matrix(output, target, num_classes=output.shape[1]).cpu().detach().numpy()))
        ...
        self.training_step_outputs.clear()  # free memory


