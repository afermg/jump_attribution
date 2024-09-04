import lightning as L

import os
import numpy as np
import pandas as pd
from more_itertools import unzip
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.nn.functional import cross_entropy, l1_loss


## WORK WITH TORCH METRIC FROM LIGHTNING INSTEAD
from torchmetrics.classification import (
MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix)



class LightningModel(L.LightningModule):
    def __init__(self, model, model_param, lr, weight_decay, max_epoch, n_class, apply_softmax=True):
        super().__init__()
        self.model = model(*model_param)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.n_class = n_class
        self.apply_softmax = apply_softmax
        self.save_hyperparameters(ignore="model")
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.accuracy = nn.ModuleList([MulticlassAccuracy(num_classes=self.n_class, average="macro") for _ in range(2)])
        self.rocauc = nn.ModuleList([MulticlassAUROC(num_classes=self.n_class, average="macro") for _ in range(2)])
        self.f1 = nn.ModuleList([MulticlassF1Score(num_classes=self.n_class, average="macro") for _ in range(2)])
        self.prefix_to_id = {"train": 0, "val": 1}

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)
        self.validation_step_outputs.append((output, target))

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #We use sync_dist to aggregate the loss accross all workers.
        #But in the case where you aggregate the result before hand with all_gather and perform some operation there is no need
        #to use sync_dist. Instead we use self.trainer.is_global_zero: and then rank_zero_only=true


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_train_epoch_end(self):
        prefix = "train"
        self._shared_eval(self.training_step_outputs, prefix)
        self.training_step_outputs.clear() # free memory
        self.training_step_outputs = []
        self.accuracy[self.prefix_to_id[prefix]].reset()
        self.rocauc[self.prefix_to_id[prefix]].reset()
        self.f1[self.prefix_to_id[prefix]].reset()


    def on_validation_epoch_end(self):
        prefix = "val"
        self._shared_eval(self.validation_step_outputs, prefix)
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_outputs = []
        self.accuracy[self.prefix_to_id[prefix]].reset()
        self.rocauc[self.prefix_to_id[prefix]].reset()
        self.f1[self.prefix_to_id[prefix]].reset()

    def _shared_eval(self, prefix_step_outputs, prefix):
        all_preds, all_labels = map(lambda x: list(x), unzip(prefix_step_outputs))
        (all_preds, all_labels) = (self.all_gather(torch.vstack(all_preds)).view(-1, self.n_class), 
                                   self.all_gather(torch.hstack(all_labels)).view(-1))

        if self.apply_softmax:
            all_preds = nn.Softmax(dim=1)(all_preds)
        
        if self.trainer.is_global_zero:
            self.log(prefix + "_" + "acc", self.accuracy[self.prefix_to_id[prefix]](all_preds, all_labels),
                     rank_zero_only=True)
            self.log(prefix + "_" + "roc", self.rocauc[self.prefix_to_id[prefix]](all_preds, all_labels),
                     rank_zero_only=True)
            self.log(prefix + "_" + "f1", self.f1[self.prefix_to_id[prefix]](all_preds, all_labels),
                     rank_zero_only=True)
                             
            # if self.current_epoch==self.max_epoch-1:
            #     self.log(prefix + "_ConfusionMatrix", 
            #              MulticlassConfusionMatrix(num_classes=self.n_class)(all_preds, all_labels), 
            #              rank_zero_only=True,
            #              sync_dist=True)

# This model needs to be tested, apparently all gather is already done under the hood with MulticlassAccuracy so no need to do things manually!

class LightningModelV2(L.LightningModule):
    def __init__(self, model, model_param, lr, weight_decay, max_epoch, n_class, apply_softmax=True):
        super().__init__()
        self.model = model(*model_param)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.n_class = n_class
        self.apply_softmax = apply_softmax
        self.save_hyperparameters(ignore="model")

        # Define metrics once
        self.train_accuracy = MulticlassAccuracy(num_classes=self.n_class, average="macro")
        self.val_accuracy = MulticlassAccuracy(num_classes=self.n_class, average="macro")
        self.train_rocauc = MulticlassAUROC(num_classes=self.n_class, average="macro")
        self.val_rocauc = MulticlassAUROC(num_classes=self.n_class, average="macro")
        self.train_f1 = MulticlassF1Score(num_classes=self.n_class, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=self.n_class, average="macro")

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)

        # Apply softmax if needed
        if self.apply_softmax:
            output = nn.Softmax(dim=1)(output)

        # Update and log metrics
        self.train_accuracy.update(output, target)
        self.train_rocauc.update(output, target)
        self.train_f1.update(output, target)

        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = cross_entropy(output, target)

        # Apply softmax if needed
        if self.apply_softmax:
            output = nn.Softmax(dim=1)(output)

        # Update and log metrics
        self.val_accuracy.update(output, target)
        self.val_rocauc.update(output, target)
        self.val_f1.update(output, target)

        # Log the loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        # Log the metric objects (this computes the final value)
        self.log("train_acc", self.train_accuracy, on_epoch=True, sync_dist=True)
        self.log("train_roc", self.train_rocauc, on_epoch=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_epoch=True, sync_dist=True)

        # Reset metrics for the next epoch
        self.train_accuracy.reset()
        self.train_rocauc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        # Log the metric objects (this computes the final value)
        self.log("val_acc", self.val_accuracy, on_epoch=True, sync_dist=True)
        self.log("val_roc", self.val_rocauc, on_epoch=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True)

        # Reset metrics for the next epoch
        self.val_accuracy.reset()
        self.val_rocauc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer



    
class LightningGAN(pl.LightningModule):
    def __init__(
            self,
            inner_generator,
            style_encoder,
            generator,
            discriminator,
            inner_generator_param,
            style_encoder_param,
            discriminator_param,
            adam_param_g,
            adam_param_d,
            beta_moving_avg,
            if_reshape_vector=False,
            H_target_shape=None

    ):
        super().__init__()
        self.save_hyperparameters(ignore=["inner_generator", "style_encoder", "generator", "discriminator"])
        #Allow a manual optimization which is required for GANs
        self.automatic_optimization = False

        # Networks
        self.generator = generator(inner_generator(*inner_generator_param), style_encoder(*style_encoder_param))
        self.discriminator = discriminator(*discriminator_param)
        self.generator_ema = generator(inner_generator(*inner_generator_param), style_encoder(*style_encoder_param))
        self.copy_parameters(self.generator, self.generator_ema)

        # Optimizer
        self.adam_param_g = adam_param_g
        self.adam_param_d = adam_param_d

        # Exponential Moving Average to stabilize the training of GANs
        self.beta_moving_avg = beta_moving_avg
        # Wether to reshape the vector or not if it is an image already
        self.if_reshape_vector = if_reshape_vector
        self.H_target_shape = H_target_shape

    def cycle_loss(self, x, x_cycle):
        return l1_loss(x, x_cycle)

    def adversarial_loss(self, y_hat, y):
        return cross_entropy(y_hat, y)

    def reshape_vector(self, x):
        """
        Expect x to have the following dimension 1D: N_features
        """
        return x.reshape(1, self.H_target_shape, -1)
##### NB, these moving average over parameters and copy of parameters only average weights and bias. It does not take into account
##### Batch Normalization layers ! For a more general implementation, maybe it would be interesting to implement this:
##### https://github.com/yasinyazici/EMA_GAN/blob/master/common/misc.py#L64
##### from the paper THE UNUSUAL EFFECTIVENESS OF AVERAGING IN GAN TRAINING Yasin Yazıcı et al. 2019
    def exponential_moving_average(self, model, ema_model):
    """Update the EMA model's parameters with an exponential moving average"""
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            # in place modif of ema_param with : ema_param * 0.999 + 0.001 * param
            ema_param.data.mul_(self.beta_moving_avg).add_((1 - self.beta_moving_avg) * param.data)

    def copy_parameters(self, source_model, target_model):
    """Copy the parameters of a model to another model"""
        for param, target_param in zip(source_model.parameters(), target_model.parameters()):
            target_param.data.copy_(param.data)
#####
    def training_step(self, batch):
        x, y = batch
        # Create target and style image
        random_index = torch.randperm(len(y))
        x_style = x[random_index].clone()
        y_target = y[random_index].clone()

        # Fetch the optimizers
        optimizer_g, optimizer_d = self.optimizers()
        # Train generator
        # Disable grad
        self.toggle_optimizer(optimizer_g) #Require grad
        self.untoggle_optimizer(optimizer_d) #Disable grad

        # Generate images
        x_fake = self.generator(x, x_style)
        x_cycle = self.generator(x_fake, x)
        # Discriminate
        discriminator_x_fake = self.discriminator(x_fake)

        # Losses to  train the generator
        # 1. make sure the image can be reconstructed
        cycle_loss = self.cycle_loss(x, x_cycled)
        # 2. make sure the discriminator is fooled
        adv_loss = self.adversarial_loss(discriminator_x_fake, y_target)
        # Total g_loss
        g_loss = cycle_loss + adv_loss
        # Optimize the generator
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        # Train Discriminator
        self.toggle_optimizer(optimizer_d) #Require grad
        self.untoggle_optimizer(optimizer_g) #Disable grad

        discriminator_x = discriminator(x)
        discriminator_x_fake = discriminator(x_fake.detach())
        # Losses to train the discriminator
        # 1. make sure the discriminator can tell real is real
        real_loss = self.adversarial_loss(discriminator_x, y)
        # 2. make sure the discriminator can tell fake is fake
        fake_loss = -self.adversarial_loss(discriminator_x_fake, y_target)
        # Total d_loss
        d_loss = (real_loss + fake_loss) * 0.5
        # Optimize the discriminator
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        # Logging every loss.
        self.log("cycle_loss", cycle_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("adv_loss", adv_loss, on_epoch=True, logger=True, sync_dist=True)
        self.log("d_loss", d_loss, on_epoch=True, logger=True, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.log_images_with_colormap(x, x_style, x_fake, x_cycled, y, y_target, idx=0)

        self.exponential_moving_average(self.generator, self.generator_ema)

    def on_train_epoch_end(self):
        self.copy_parameters(self.generator_ema, self.generator)


    def log_images_with_colormap(self, x, x_style, x_fake, x_cycled, y, y_target idx=0):
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        # Convert tensors to NumPy
        input_img = x[idx].cpu().detach().numpy()
        style_img = x_style[idx].cpu().detach().numpy()
        fake_img = x_fake[idx].cpu().detach().numpy()
        cycled_img = x_cycled[idx].cpu().detach().numpy()
        input_class = y[idx].cpu().detach().numpy()
        style_class = y_target[idx].cpu().detach().numpy()
        if self.if_reshape_vector:
            input_img = self.reshape_vector(input_img)
            style_img = self.reshape_vector(style_img)
            fake_img = self.reshape_vector(fake_img)
            cycled_img = self.reshape_vector(cycled_img)

        axs[0].imshow(input_img.permute(1, 2, 0).detach().numpy())
        axs[0].set_title(f"Input image - Class_{input_class}")
        axs[1].imshow(style_img.permute(1, 2, 0).detach().numpy())
        axs[1].set_title(f"Style image - Class_{style_class}")
        axs[2].imshow(fake_img.permute(1, 2, 0).detach().numpy())
        axs[2].set_title(f"Generated image - Class_{style_class}")
        axs[3].imshow(cycled_img.permute(1, 2, 0).detach().numpy())
        axs[3].set_title(f"Cycled image - Class_{input_class}")

        for ax in axs:
            ax.axis("off")
        # Save the figure as a TensorBoard image
        self.logger.experiment.add_figure("train/generated_images", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), **self.adam_param_g)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), **self.adam_param_d)
        return [opt_g, opt_d], []

