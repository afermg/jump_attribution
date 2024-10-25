#!/usr/bin/env python
# coding: utf-8
from itertools import starmap
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from captum.attr import Saliency, IntegratedGradients, DeepLift, GuidedBackprop, GuidedGradCam, LayerGradCam, LayerActivation
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric

torch.manual_seed(42)
np.random.seed(42)

# General Attribution function
def Attribution(model: nn.Module,
                method: Callable,
                inputs: Union[Tensor, Tuple[Tensor, ...]],
                baselines: BaselineType = None,
                inputs_target: TargetType = None,
                method_kwargs: Optional[Dict[str, Any]] = None,
                attr_kwargs: Optional[Dict[str, Any]] = None,
                ):
    grad_func = method(model, **method_kwargs)
    return grad_func.attribute(inputs=inputs, baselines=baselines, target=inputs_target, **attr_kwargs)

# D_INGRADS (Input * Gradient)
#
class D_INGRADS(Saliency):
    """
    -- Updated discriminatory method of the InputXGradient method --
    A baseline approach for computing the attribution. It multiplies input with
    the gradient with respect to input.
    https://arxiv.org/abs/1605.01713
    """
    def __init__(self, model: Callable[..., Tensor]) -> None:
        """
        Args:

            model (Callable): The forward function of the model or any
                          modification of it
        """
        super().__init__(model)

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: object = None) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.

            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                            Baselines define the point from which we differentiate with inputs.
                            and can be provided as:

                            - a single tensor, if inputs is a single tensor, with
                              exactly the same dimensions as inputs or the first
                              dimension is one and the remaining dimensions match
                              with inputs.

                            - a single scalar, if inputs is a single tensor, which will
                              be broadcasted for each input value in input tensor.

                            - a tuple of tensors or scalars, the baseline corresponding
                              to each tensor in the inputs' tuple can be:

                              - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.

                              - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.

                            In the cases when `baselines` is not provided, we internally
                            use zero scalar corresponding to each input tensor.

            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None

            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None

        Returns:
                *Tensor* or *tuple[Tensor, ...]* of **attributions**:
                - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                            The input x gradient with
                            respect to each input feature. Attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # Generating random input with size 2x3x3x32
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Defining InputXGradient interpreter
            >>> input_x_gradient = InputXGradient(net)
            >>> # Computes inputXgradient for class 4.
            >>> attribution = input_x_gradient.attribute(input, baselines, target=4)
        """

        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, formatted_baselines = _format_input_baseline(inputs, baselines)
        # Call Saliency's attribute method to get the gradients
        saliency = super().attribute(inputs=inputs, target=target, additional_forward_args=additional_forward_args)
        # Modify the gradients as per your original D_INGRADS function
        attributions = tuple(starmap(
            lambda ingrads, inputs, baselines: torch.abs(ingrads * (inputs - baselines)),
            zip(saliency, formatted_inputs, formatted_baselines)))

        return  _format_output(is_inputs_tuple, attributions)

# D_IG (Integrated Gradient)
# Already implemented, call IntegratedGradients

# D_DL (DeepLift)
# Already implemented, call DeepLift

# D_GC (GradCAM)

class D_GradCam(LayerGradCam):
    """
    Computes GradCAM attribution for last layer. GradCAM is designed for
    convolutional neural networks, and is usually applied to the last
    convolutional layer.

    GradCAM computes the gradients of the target output with respect to
    the given layer, averages for each output channel (dimension 2 of
    output), and multiplies the average gradient for each channel by the
    layer activations. The results are summed over all channels.

    Note that in the original GradCAM algorithm described in the paper,
    ReLU is applied to the output, returning only non-negative attributions.
    For providing more flexibility to the user, we choose to not perform the
    ReLU internally by default and return the sign information. To match the
    original GradCAM algorithm, it is necessary to pass the parameter
    relu_attributions=True to apply ReLU on the final
    attributions or alternatively only visualize the positive attributions.

    Note: this procedure sums over the second dimension (# of channels),
    so the output of GradCAM attributions will have a second
    dimension of 1, but all other dimensions will match that of the layer
    output.

    GradCAM attributions are generally upsampled and can be viewed as a
    mask to the input, since a convolutional layer output generally
    matches the input image spatially. This upsampling can be performed
    using LayerAttribution.interpolate, as shown in the example below.

    More details regarding the GradCAM method can be found in the
    original paper here:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module, num_layer: int = -1):
        """
        Args:

            model (nn.Module): The  model for which to compute attribution. Must contain a convolution layer.
            num_layer (Optional, int): the conv layer number considered.
            Default: -1 # the last one as being the recommended workflow for GradCAM

        """
        try:
            layer_name, layer = [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)][num_layer]
            # define self.forward_func, self.layer, self.device_ids
            super().__init__(model, layer)
        except:
            raise Exception(f"model should contains a 'torch.nn.Conv2d'. Here there is only: {set(map(lambda x: type(x), model.modules()))}")

    def attribute(self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        average_grad_channels: bool = True,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        attr_dim_summation: bool = True,
        attr_interpolate: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:


        """
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
        baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            average_grad_channels (bool, optional): Indicate whether to
                        average gradient of across channels before multiplying by the
                        feature map. The default is set to true as it is the default
                        GradCAM behavior.
                        Default: True
            relu_attributions (bool, optional): Indicates whether to
                        apply a ReLU operation on the final attribution,
                        returning only non-negative attributions. Setting this
                        flag to True matches the original GradCAM algorithm,
                        otherwise, by default, both positive and negative
                        attributions are returned.
                        Default: False
            attr_dim_summation (bool, optional): Indicates whether to
                        sum attributions along dimension 1 (usually channel).
                        The default (True) means to sum along dimension 1.
                        Default: True
            attr_interpolate (bool, optional): Indicates whether to interpolate the
                        attribution so it match input dim.
                        Default: True

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx50x8x8.
            >>> # It is the last convolution layer, which is the recommended
            >>> # use case for GradCAM.
            >>> net = ImageClassifier()
            >>> layer_gc = D_GradCam(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer GradCAM for class 3 relative to baseline.
            >>> # attribution size matches layer output except for dimension
            >>> # 1, so dimensions of attr would be Nx1x8x8.
            >>> attr = layer_gc.attribute(input, baseline,  3)
            >>> # GradCAM attributions are often upsampled and viewed as a
            >>> # mask to the input, since the convolutional layer output
            >>> # spatially matches the original input image.
            >>> # This can be done with LayerAttribution's interpolate method.
            >>> # This is the default behavior but it can be cancelled.
            >>> upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))

        """

        inputs, baselines = _format_input_baseline(inputs, baselines)
        # baseline must be turn into a tuple of tensor to compute activation
        baselines = tuple(starmap(
            lambda baseline, _input: (torch.tensor(baseline,
                                                   dtype=torch.float32,
                                                   device=_input.device).expand(_input.size()[1:]).unsqueeze(0)
                                      if isinstance(baseline, (int, float)) else baseline),
            zip(baselines, inputs)
        ))

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )

        layer_eval_baselines = tuple(
            map(lambda baseline:
                (LayerActivation(self.forward_func, self.layer, self.device_ids)
                 .attribute(baseline,
                            additional_forward_args,
                            attribute_to_layer_input=attribute_to_layer_input)),
                baselines))


        summed_grads = tuple(
            map(lambda layer_grad:
                (torch.mean(layer_grad,
                            dim=tuple(x for x in range(2, len(layer_grad.shape))),
                            keepdim=True)
                    if (len(layer_grad.shape) > 2 and average_grad_channels)
                    else layer_grad
                 ),
                layer_gradients))


        scaled_acts = tuple(
            starmap(lambda summed_grad, layer_eval, layer_eval_baseline:
                    (torch.sum(summed_grad * (layer_eval - layer_eval_baseline), dim=1, keepdim=True)
                     if attr_dim_summation
                     else summed_grad * (layer_eval - layer_eval_baseline)),
                    zip(summed_grads, layer_evals, layer_eval_baselines)
                    ))

        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        #use inheriting method from LayerAttribution (interpolate and take the absolute value of the projection)
        scaled_acts = tuple(
            starmap(lambda scaled_act, _input:
            (torch.abs(self.interpolate(layer_attribution=scaled_act,
                              interpolate_dims=_input.size()[2:],
                              interpolate_mode="bilinear")) #default being 'nearest':  'bilinear' provides smoother results which is desirable for large image.
             if (len(_input.size()) > 2 and attr_interpolate)
             else scaled_act),
             zip(scaled_acts, inputs)
                    ))

        return _format_output(len(scaled_acts) > 1, scaled_acts)



# D_GGC (GuidedGradCAM)

# Random

# Residual

"""
-----------------------------------------------------------
"""


import zarr

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
    X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
    X_real.requires_grad, X_fake.requires_grad = True, True
    # with torch.no_grad():
    #     y_hat_real = F.softmax(VGG_model(X_real), dim=1).argmax(dim=1)
    #     y_hat_fake = F.softmax(VGG_model(X_fake), dim=1).argmax(dim=1)
    saliency = Attribution(VGG_model, D_GradCam, X_real, X_fake, y_fake) # add method_kwargs, or attr_kwargs depending on the method.
    break
print(saliency)

# layer_name, layer = next((name, module) for name, module in reversed(list(VGG_model.named_modules())) if isinstance(module, torch.nn.Conv2d))

# from itertools import starmap
# VGG_model.zero_grad()
# grads, evals = map(lambda x: x[0], compute_layer_gradients_and_eval(VGG_model,
#                                                                     layer,
#                                                                     X_real,
#                                                                     y_real
#                                                                     ))
# test_grad = LayerGradCam(VGG_model, layer).attribute(X_real, y_real , relu_attributions=False, attr_dim_summation=False)

# layer_gradients, layer_evals = compute_layer_gradients_and_eval(VGG_model,
#                                                                 layer,
#                                                                 X_real,
#                                                                 y_real
#                                                                 )

# summed_grads = tuple(
#             (
#                 torch.mean(
#                     # pyre-fixme[6]: For 1st argument expected `Tensor` but got
#                     #  `Tuple[Tensor, ...]`.
#                     layer_grad,
#                     # pyre-fixme[16]: `tuple` has no attribute `shape`.
#                     dim=tuple(x for x in range(2, len(layer_grad.shape))),
#                     keepdim=True,
#                 )
#                 if len(layer_grad.shape) > 2
#                 else layer_grad
#             )
#             for layer_grad in layer_gradients
#         )

# layer_name, layer = next((name, module) for name, module in reversed(list(VGG_model.named_modules())) if isinstance(module, torch.nn.Conv2d))

# layer_gradients, layer_evals = compute_layer_gradients_and_eval(
#     VGG_model,
#     layer,
#     X_real,
#     y_real)
# layer_evals2 = LayerActivation(VGG_model, layer).attribute(X_real)

# baselines = (0, )
# inputs = (X_real, )
# baselines = tuple(starmap(
#     lambda baseline, _input: (torch.tensor(baseline, dtype=torch.float32, device=_input.device).expand(_input.size()[1:]).unsqueeze(0)
#                               if isinstance(baseline, (int, float)) else baseline),
#     zip(baselines, inputs)
# ))
