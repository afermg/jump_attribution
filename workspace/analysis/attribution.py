#!/usr/bin/env python
# coding: utf-8
from functools import partial
from itertools import starmap

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_vis
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    IntegratedGradients,
    LayerActivation,
    LayerGradCam,
    Saliency,
)
from captum.attr import visualization as viz
from captum.attr._utils.common import (
    _format_input_baseline,
)
from torch import Tensor
from torch.nn import Module

torch.manual_seed(42)
np.random.seed(42)
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import pandas as pd
import seaborn as sns
from tqdm import tqdm

"""
------- Attribution method ----------
"""
# General Attribution function
def Attribution(model: nn.Module,
                method: Callable,
                inputs: Union[Tensor, Tuple[Tensor, ...]],
                baselines: BaselineType = None,
                inputs_target: TargetType = None,
                baselines_target: TargetType = None,
                method_kwargs: Optional[Dict[str, Any]] = None,
                attr_kwargs: Optional[Dict[str, Any]] = None,
                ):
    if method_kwargs is not None:
        grad_func = method(model, **method_kwargs)
    else:
        grad_func = method(model)
    if attr_kwargs is not None:
        return grad_func.attribute(inputs=inputs, baselines=baselines, target=inputs_target, **attr_kwargs)
    else:
        return grad_func.attribute(inputs=inputs, baselines=baselines, target=inputs_target)

def build_kwargs_dict(attr_names: List[str],
                      kwargs_dict: Optional[Dict[Dict[str, Any], Any]] = None):
    """
    Build kwargs_dict for Attribution function.
    attr_names (list of string): attr_names is intended as a list of name associated to the attribution method:
    kwargs_dict (optional dict of dict): An example of correct method_kwargs_dict could be {"D_GradCam": {"num_layer": -3}}
    """
    if kwargs_dict is None:
        return {attr_name: None for attr_name in attr_names}
    else:
        attr_names_missing = set(attr_names) - set(kwargs_dict.keys())
        kwargs_dict.update({attr_name: None for attr_name in attr_names_missing})
        return kwargs_dict


# D_InGrad (Input * Gradient)
#
class D_InGrad(Saliency):
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
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores, D_InGrads
                        computes the differences between the inputs and
                        corresponding references.

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
        # Modify the gradients as per your original D_InGrad function
        attributions = tuple(starmap(
            lambda ingrads, inputs, baselines: torch.abs(ingrads * (inputs - baselines)),
            zip(saliency, formatted_inputs, formatted_baselines)))

        return  _format_output(is_inputs_tuple, attributions)

# D_IG (Integrated Gradient)
# Already implemented, call IntegratedGradients

# D_DL (DeepLift)
# Already implemented, call DeepLift

# Careful: Need to redefine relu in your neural network every time so it work. So instead of defining self.relu = nn.ReLU once,
# do nn.ReLU every time.

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
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores, D_GradCam
                        computes the differences between the inputs/outputs and
                        corresponding references.

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

class D_GuidedGradCam(D_GradCam):
    def __init__(self, model: nn.Module, num_layer: int = -1):
        """
        Args:

            model (nn.Module): The  model for which to compute attribution. Must contain a convolution layer.
            num_layer (Optional, int): the conv layer number considered.
            Default: -1 # the last one as being the recommended workflow for GradCAM

        """
        super().__init__(model, num_layer)
        self.guided_backprop = GuidedBackprop(model)

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
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores D_GuidedGradCam
                        computes the differences between the inputs/outputs and
                        corresponding references.
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
            >>> layer_gc = D_GuidedGradCam(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer GradCAM for class 3 relative to baseline.
            >>> # attribution size matches layer output except for dimension
            >>> # 1, so dimensions of attr would be Nx1x8x8.
            >>> attr = layer_gc.attribute(input, baseline,  3)
            >>> # Then GradCAM attributions are upsampled and multiplied with
            >>> # gradient of the input.
            >>> # The resulted attributions spatially matches the original input image.

        """

        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, formatted_baselines = _format_input_baseline(inputs, baselines)
        # Call Saliency's attribute method to get the gradients
        d_gcs = super().attribute(inputs,
                                  baselines,
                                  target,
                                  additional_forward_args,
                                  average_grad_channels,
                                  attribute_to_layer_input,
                                  relu_attributions,
                                  attr_dim_summation,
                                  attr_interpolate)

        # Modify the gradients as per your original D_InGrad function
        attributions = tuple(starmap(
            lambda d_gc, _input: d_gc * self.guided_backprop.attribute.__wrapped__(
                self.guided_backprop, inputs, target, additional_forward_args),
            zip(d_gcs, formatted_inputs)))

        return  _format_output(is_inputs_tuple, attributions)

# Random

class Random_attr():
    """
    Compute random attribution attribution map.
    """
    def __init__(self, *args):
        pass
    def attribute(self,
                  inputs: Union[Tensor, Tuple[Tensor, ...]],
                  *args,
                  **kwargs):
        """
            inputs (Tensor or tuple[Tensor, ...]): Input for which the random attribution
                   is computed.
        """

        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs = _format_tensor_into_tuples(inputs)
        # apply a gaussian blur with sigma = 4.0 and kernel which match definition of scipy:
        # kernel_size = round(2 * radius + 1) where radius = truncated * sigma where truncated = 4.0 for default.
        attributions = tuple(map(
            lambda _input: F_vis.gaussian_blur(
                torch.abs(torch.randn(*_input.shape)),
                kernel_size=round(2 * round(4.0 * 4.0) + 1), sigma=4.0),
           formatted_inputs))

        attributions = tuple(map(
            lambda attr: (attr - attr.min())/(attr.max() - attr.min()),
            attributions))
        return  _format_output(is_inputs_tuple, attributions)

# Residual

class Residual_attr():
    """
    Compute the difference between inputs and baselines and apply a MinMaxScaler
    """
    def __init__(self, *args):
        pass
    def attribute(self,
                  inputs: Union[Tensor, Tuple[Tensor, ...]],
                  baselines: BaselineType = None,
                  *args,
                  **kwargs):
        """
            inputs (Tensor or tuple[Tensor, ...]): Input for which attributions
                        are computed.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores, Residual_attr
                        computes the differences between the inputs and
                        corresponding references.

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
        """
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, formatted_baselines = _format_input_baseline(inputs, baselines)
        attributions = tuple(starmap(
            lambda _input, baseline: torch.abs(_input - baseline),
            zip(formatted_inputs, formatted_baselines)))
        attributions = tuple(map(
            lambda attr: (attr - attr.min()) / (attr.max() - attr.min()),
            attributions))
        return  _format_output(is_inputs_tuple, attributions)



@torch.no_grad
def normalize_attribution(attribution: torch.Tensor, percentile: Union[int, float]=98) -> torch.Tensor:
    # Sum over channels (c dimension) in place
    attribution = torch.sum(attribution, dim=1) # keepdim=True) # Shape: (n, k, l)

    # Flatten, take absolute value, and sort in place
    sorted_vals, _ = torch.sort(torch.abs(attribution).view(-1))  # Sorts in ascending order
    cum_sums = torch.cumsum(sorted_vals, dim=0)
    threshold_idx = torch.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    threshold = sorted_vals[threshold_idx].item()  # Extracts the threshold

    # Normalize in place and clamp between -1 and 1
    if abs(threshold) < 1e-5:
        warnings.warn(
            "Attempting to normalize by a value approximately 0; visualized results may be misleading. "
            "This likely means that attribution values are all close to 0.",
            stacklevel=2,
        )
    attribution.div_(threshold).clamp_(-1, 1)
    return attribution

"""
------- Visualisation of attribution function ----------
"""

def visualize_attribution(attributions, X_real, X_fake, y_real, y_fake, y_hat_real, y_hat_fake,  method_names=None,
                          fig=None, axes=None):
    # Ensure X_real and X_fake are in channel-last format for plotting
    X_real = np.transpose(X_real, (1, 2, 0))
    X_fake = np.transpose(X_fake, (1, 2, 0))

    num_methods = len(attributions)  # Number of attribution methods
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, num_methods + 2, figsize=(5 * (num_methods + 2), 5))

    # Plot X_real once in the first column
    axes[0].imshow(X_real)
    axes[0].set_title(f"Original Image\nclass: {y_real} / pred: {y_hat_real}")
    axes[0].axis("off")

    # Plot X_fake once in the second column
    axes[1].imshow(X_fake)
    axes[1].set_title(f"Counterfactual Image\nclass: {y_fake} / pred: {y_hat_fake}")
    axes[1].axis("off")

    # Plot each attribution map in subsequent columns as a heatmap
    for i, attribution in enumerate(attributions):
        im = axes[i + 2].imshow(attribution, cmap="bwr_r", vmin=-1, vmax=1) # attribution should be already normalized
        title = method_names[i] if method_names else f"Attribution {i+1}"
        axes[i + 2].set_title(title)
        axes[i + 2].axis("off")

    # Add color bar for the attribution maps
    cbar = fig.colorbar(im, ax=axes[-1], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label("Attribution Score")

    # plt.tight_layout()
    return fig, axes

def visualize_attribution_mask(attributions, mask_weight, mask_size,
                               X_real, X_fake, y_real, y_fake, y_hat_real, y_hat_fake,
                               method_names=None,
                               fig=None, axes=None):
    # Ensure X_real and X_fake are in channel-last format for plotting
    X_real = np.transpose(X_real, (1, 2, 0))
    X_fake = np.transpose(X_fake, (1, 2, 0))

    num_methods = len(attributions)  # Number of attribution methods
    num_mask = mask_size.shape[-1]
    if fig is None or axes is None:
        fig, axes = plt.subplots(num_mask + 1, num_methods + 2, figsize=(5 * (num_methods + 2), 5))

    # Plot X_real once in the first column
    axes[0][0].imshow(X_real)
    axes[0][0].set_title(f"Original Image\nclass: {y_real} / pred: {y_hat_real}")
    axes[0][0].axis("off")

    # Plot X_fake once in the second column
    axes[0][1].imshow(X_fake)
    axes[0][1].set_title(f"Counterfactual Image\nclass: {y_fake} / pred: {y_hat_fake}")
    axes[0][1].axis("off")

    # Plot each attribution map in subsequent columns as a heatmap
    for i, attribution in enumerate(attributions):
        im = axes[0][i + 2].imshow(attribution, cmap="bwr_r", vmin=-1, vmax=1) # attribution should be already normalized
        title = method_names[i] if method_names else f"Attribution {i+1}"
        axes[0][i + 2].set_title(title)
        axes[0][i + 2].axis("off")
        for j in range(num_mask):
            axes[j+1][0].axis("off")
            axes[j+1][1].axis("off")
            axes[j+1][i + 2].imshow(mask_weight[i][j], cmap="viridis", vmin=0, vmax=1)
            title = f"mask_size: {mask_size[i][j]}"
            axes[j+1][i + 2].set_title(title)
            axes[j+1][i + 2].axis("off")

    # Add color bar for the attribution maps
    cbar = fig.colorbar(im, ax=axes[0,-1], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label("Attribution Score")

    # plt.tight_layout()
    return fig, axes


def plot_attr_img(model, dataloader_real_fake, fig_directory, name_fig, num_img=24,
                  attr_methods=[D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam],
                  attr_names=["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam"],
                  percentile=98,
                  method_kwargs_dict=None,
                  attr_kwargs_dict=None):

    method_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=method_kwargs_dict)
    attr_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=attr_kwargs_dict)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    curr_img = 0
    fig, axis = plt.subplots(num_img, len(attr_methods) + 2, figsize=(5 * (len(attr_methods) + 2), 5 * num_img))
    for X_real, X_fake, y_real, y_fake in dataloader_real_fake:
        X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
        X_real.requires_grad, X_fake.requires_grad = True, True
        with torch.no_grad():
            y_hat_real = F.softmax(model(X_real), dim=1).argmax(dim=1)
            y_hat_fake = F.softmax(model(X_fake), dim=1).argmax(dim=1)
        attributions = []
        for attr_name, attr_method in zip(attr_names, attr_methods):
            # add method_kwargs, or attr_kwargs depending on the method, # method_kwargs={"num_layer": -3})
            attr_batch = Attribution(model, attr_method, X_fake, X_real, y_fake,
                                     method_kwargs=method_kwargs_dict[attr_name],
                                     attr_kwargs=attr_kwargs_dict[attr_name])

            attr_batch = normalize_attribution(attr_batch, percentile=percentile).detach().cpu().numpy()
            attributions.append(attr_batch)
            torch.cuda.empty_cache()

        attributions = np.stack(attributions, axis=1)
        for i in range(X_real.shape[0]):
            fig, _ = visualize_attribution(attributions[i],((1+X_real[i])/2).detach().cpu().numpy(), ((1+X_fake[i])/2).detach().cpu().numpy(),
                                              y_real[i].detach().cpu().numpy(), y_fake[i].detach().cpu().numpy(),
                                              y_hat_real[i].detach().cpu().numpy(), y_hat_fake[i].detach().cpu().numpy(),
                                              attr_names,
                                              fig, axis[curr_img, :])
            curr_img += 1
            if curr_img == num_img:
                break
        if curr_img == num_img:
            break
    fig.savefig(fig_directory / name_fig)

def plot_attr_mask_img(model, dataloader_real_fake, fig_directory, name_fig, num_img=24,
                       steps=200, selected_mask=[0, 50, 100],
                       attr_methods=[D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam],
                       attr_names=["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam"],
                       percentile=98,
                       method_kwargs_dict=None,
                       attr_kwargs_dict=None):

    method_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=method_kwargs_dict)
    attr_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=attr_kwargs_dict)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    curr_img = 0
    num_y_ax_per_img = (len(selected_mask)+1)
    fig, axis = plt.subplots(num_img * num_y_ax_per_img, len(attr_methods) + 2,
                             figsize=(5 * (len(attr_methods) + 2), 5 * num_img * (len(selected_mask)+1)))
    for X_real, X_fake, y_real, y_fake in dataloader_real_fake:
        X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
        X_real.requires_grad, X_fake.requires_grad = True, True
        with torch.no_grad():
            y_hat_real = F.softmax(model(X_real), dim=1).argmax(dim=1)
            y_hat_fake = F.softmax(model(X_fake), dim=1).argmax(dim=1)
        attributions = []
        mask_weight_selected = []
        mask_size_selected = []
        for attr_name, attr_method in zip(attr_names, attr_methods):
            # add method_kwargs, or attr_kwargs depending on the method, # method_kwargs={"num_layer": -3})
            attr_batch = Attribution(model, attr_method, X_fake, X_real, y_fake,
                                     method_kwargs=method_kwargs_dict[attr_name],
                                     attr_kwargs=attr_kwargs_dict[attr_name])

            attr_batch = normalize_attribution(attr_batch, percentile=percentile).detach().cpu().numpy()
            attributions.append(attr_batch)
            torch.cuda.empty_cache()
            with Pool(min(batch_size, cpu_count())) as pool: # with Pool(cpu_count()) as pool:
                mask_batch = pool.map(partial(get_mask, steps=steps), attributions[-1])
                mask_weight, mask_size = map(lambda l: np.array(l), list(zip(*mask_batch)))
                mask_weight_selected.append(mask_weight[:,selected_mask])
                mask_size_selected.append(mask_size[:,selected_mask])

        attributions = np.stack(attributions, axis=1)
        mask_weight_selected = np.stack(mask_weight_selected, axis=1)
        mask_size_selected = np.stack(mask_size_selected, axis=1)
        if len(mask_size_selected.shape) < 3:
            mask_weight_selected = mask_weight_selected[:, :, None]
            mask_size_selected = mask_size_selected[:, :, None]
        for i in range(X_real.shape[0]):
            fig, _ = visualize_attribution_mask(attributions[i], mask_weight_selected[i], mask_size_selected[i],
                                                ((1+X_real[i])/2).detach().cpu().numpy(), ((1+X_fake[i])/2).detach().cpu().numpy(),
                                                y_real[i].detach().cpu().numpy(), y_fake[i].detach().cpu().numpy(),
                                                y_hat_real[i].detach().cpu().numpy(), y_hat_fake[i].detach().cpu().numpy(),
                                                attr_names,
                                                fig,
                                                axis[num_y_ax_per_img * curr_img: num_y_ax_per_img * curr_img + num_y_ax_per_img,:])
            curr_img += 1
            if curr_img == num_img:
                break
        if curr_img == num_img:
            break
    fig.savefig(fig_directory / name_fig)

"""
------- Mask creation and DAC score --------
"""


def slice_array(indices, steps=1000):
    total_indices = len(indices)
    # Calculate the proportion so that the tail gather more element that the head
    proportions = np.array([100 if i < 8*steps/10 else 500 for i in range(1, steps + 1) ])
    proportions = proportions / proportions.sum()

    # Compute cumulative indices to use for slicing
    cumulative_sizes = (proportions * total_indices).cumsum().astype(int)

    # Fix rounding issues by adjusting the last index
    cumulative_sizes[-1] = total_indices

    # Use slicing based on cumulative indices
    split_arrays = [indices[cumulative_sizes[i-1] if i > 0 else 0 : cumulative_sizes[i]] for i in range(steps)]
    return split_arrays

def get_mask(attribution, steps=1000, sigma=11, struc=10):
    """
    attribution should be normalized between -1 and 1 already
    """
    if len(attribution.shape) >2:
        if len(attribution.shape) > 3:
            raise Exception("This function only work with single attribution map.")
        else:
            attribution = attribution.sum(axis=0)
    # attribution = attribution / np.abs(attribution).max()
    indices_sort = np.dstack(np.unravel_index(np.argsort(attribution.ravel()), (448, 448))).squeeze()[::-1]
    sliced_indices = slice_array(indices_sort, steps=steps)
    mask_sort = np.zeros((448, 448), dtype=np.uint8)
    mask_size_tot, mask_weight_tot = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
    for _slice in sliced_indices:
        mask_sort[tuple(_slice.T)] = 1
        mask = cv2.morphologyEx(mask_sort, cv2.MORPH_CLOSE, kernel)
        mask_size_tot.append(np.sum(mask))
        mask_weight_tot.append(cv2.GaussianBlur(mask.astype(float), (sigma,sigma),0))
    return np.stack(mask_weight_tot), np.stack(mask_size_tot)

def dac_curve_computation(model, dataloader_real_fake,
                          attr_methods=[D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam],
                          attr_names=["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam"],
                          batch_size_mask=512,
                          steps=200, shift=0.7, head_tail=(1, 5),
                          percentile=98,
                          size_closing=10,
                          size_gaussblur=11,
                          early_stop: Optional[int]=None,
                          method_kwargs_dict=None,
                          attr_kwargs_dict=None):

    def update_mat_count(indices_0: Tensor,
                         indices_1: Tensor,
                         size: int) -> Tensor:
        """
        Count the occurrence of pair of (indices_0_i, indices_1_j) at the position i,j
        of a Tensor with size size.

        Args:
            indices_0 (Tensor): 1-d int tensor
            indices_1 (Tensor): 1-d int tensor
            size (int): the size of the output matrix

        Return:
            counts (Tensor): count the occurrence of pair of (indices_0_i, indices_1_j) at the position i,j
        """
        indices = indices_0 * size + indices_1
        counts = torch.bincount(indices, minlength=size * size)
        return counts.view(size, size)

    def process_mask(mask_binary: np.ndarray,
                     kernel: np.ndarray,
                     size_gaussblur: int) -> Tuple[np.ndarray, ...]:
        """
        Return the closing of the binary mask according to the given kernel and the blurred closed mask returned by a gaussianblur filter.

        Args:
            mask_binary (np.ndarray): the binary mask, must be type np.uint8 else raise an exception
            kernel (np.ndarray): the kernel for closing operation
            size_gaussblur (int): the size of the kernel for the gaussian blur

        Return:
            mask_binary_closed (np.ndarray): the closed binary mask
            mask_weight (np.ndarray): the blurred closed binary mask
        """

        if mask_binary.dtype != np.uint8:
            raise Exception(f"morphology closing can only happen with np.uint8 type of mask. Instead mask_binary.dtype is: {mask_binary.dtype}")

        mask_binary_closed = cv2.morphologyEx(mask_binary, op=cv2.MORPH_CLOSE, kernel=kernel)
        mask_weight = cv2.GaussianBlur(mask_binary_closed.astype(np.float32), ksize=(size_gaussblur, size_gaussblur), sigmaX=0)
        return mask_binary_closed, mask_weight

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    proportions = np.array([head_tail[0] if i < shift*steps else head_tail[1] for i in range(steps)])
    tot_pixels = dataloader_real_fake.dataset[0][0].shape[-2:].numel()
    splitting_point = ((proportions * tot_pixels)/proportions.sum()).cumsum().astype(int)
    splitting_point[-1] = tot_pixels - 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_closing, size_closing))
    mask_size_tot = {attr_name: [] for attr_name in attr_names}
    dac_tot = {attr_name: [] for attr_name in attr_names}

    method_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=method_kwargs_dict)
    attr_kwargs_dict = build_kwargs_dict(attr_names=attr_names, kwargs_dict=attr_kwargs_dict)

    if early_stop is not None:
        count = 0

    for X_real, X_fake, y_real, y_fake in tqdm(dataloader_real_fake, leave=True, desc="batch"):
        X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
        with torch.no_grad():
            y_hat_real_log = F.softmax(model(X_real), dim=1)#.argmax(dim=1)
            y_hat_fake_log = F.softmax(model(X_fake), dim=1)#.argmax(dim=1)
            pred_true_mask = (y_real == y_hat_real_log.argmax(dim=1)) & (y_fake == y_hat_fake_log.argmax(dim=1))

        # compute accuracy of the classification so that diagonale element are pred of real images and the rest if correct transition
        # in such a matrix,  mat_acc[0][1] is the count of correct transition from y_real = 0 to y_fake=1
        try:
            mat_count += update_mat_count(y_real, y_real, mat_count.size(0))
            mat_count += update_mat_count(y_real, y_fake, mat_count.size(0))
            mat_acc += update_mat_count(y_real[pred_true_mask], y_real[pred_true_mask], mat_acc.size(0))
            mat_acc += update_mat_count(y_real[pred_true_mask], y_fake[pred_true_mask], mat_acc.size(0))
        except:
            mat_count = torch.zeros((y_hat_real_log.shape[1], y_hat_real_log.shape[1]), device=device)
            mat_acc = torch.zeros((y_hat_real_log.shape[1], y_hat_real_log.shape[1]), device=device)
            mat_count += update_mat_count(y_real, y_real, mat_count.size(0))
            mat_count += update_mat_count(y_real, y_fake, mat_count.size(0))
            mat_acc += update_mat_count(y_real[pred_true_mask], y_real[pred_true_mask], mat_acc.size(0))
            mat_acc += update_mat_count(y_real[pred_true_mask], y_fake[pred_true_mask], mat_acc.size(0))

        if pred_true_mask.sum() > 0:
            X_real = X_real[pred_true_mask]
            X_fake = X_fake[pred_true_mask]
            y_real = y_real[pred_true_mask]
            y_fake = y_fake[pred_true_mask]
        else:
            continue

        X_real.requires_grad, X_fake.requires_grad = True, True
        for attr_name, attr_method in zip(attr_names, attr_methods):
            # add method_kwargs, or attr_kwargs depending on the method, # method_kwargs={"num_layer": -3})
            attr_batch = Attribution(model, attr_method, X_fake, X_real, y_fake,
                                     method_kwargs=method_kwargs_dict[attr_name],
                                     attr_kwargs=attr_kwargs_dict[attr_name])
            attr_batch = normalize_attribution(attr_batch, percentile=percentile)
            attr_sorted_index = torch.dstack(torch.unravel_index(torch.argsort(attr_batch.view(attr_batch.size(0), -1), dim=1, descending=True),
                                                                 tuple(X_real.shape[-2:])))
            attr_sorted_index = attr_sorted_index[:, splitting_point]
            mask_binary_all = torch.ge(attr_batch.unsqueeze(1),
                                       attr_batch[np.arange(5)[:, None],
                                                  attr_sorted_index[...,0],
                                                  attr_sorted_index[...,1]].unsqueeze(-1).unsqueeze(-1)).cpu().numpy().astype(np.uint8)
            # free up memory
            del attr_batch, attr_sorted_index
            torch.cuda.empty_cache()

            func_process_mask = partial(process_mask, kernel=kernel, size_gaussblur=size_gaussblur)
            mask_binary_all = mask_binary_all.reshape(-1, *mask_binary_all.shape[-2:])
            with ThreadPool(cpu_count()) as pool:
                result = pool.map(func_process_mask, mask_binary_all)
            mask_binary_closed_all, mask_weight_all = tuple(map(lambda x: np.array(x).reshape(X_real.shape[0], -1, *X_real.shape[-2:]), zip(*result)))

            dac_batch = []
            mask_weight_all_split = np.array_split(mask_weight_all,
                                                   np.arange(0, mask_weight_all.shape[1], batch_size_mask // mask_weight_all.shape[0]),
                                                   axis=1)[1:]
            with torch.no_grad():
                X_real, X_fake = X_real.unsqueeze(1), X_fake.unsqueeze(1)
                for mask_weight in mask_weight_all_split:
                    mask_weight = torch.tensor(mask_weight).to(device).unsqueeze(-3)
                    X_hybrid = ((1 - mask_weight) * X_fake + mask_weight * X_real).view(-1, *X_real.shape[-3:])

                    y_hat_hybrid_log = F.softmax(model(X_hybrid), dim=1).view(*mask_weight.shape[:2], -1)
                    dac_batch.append((y_hat_hybrid_log[np.arange(len(y_real)), :, y_real] - y_hat_fake_log[np.arange(len(y_real)), y_real].unsqueeze(1)).cpu().numpy())
                # free up memory
                del mask_weight, X_hybrid, y_hat_hybrid_log
                torch.cuda.empty_cache()

            mask_size_tot[attr_name].append(mask_binary_closed_all.sum(axis=(2,3)))
            dac_tot[attr_name].append(np.column_stack(dac_batch))
        if early_stop is not None:
            count += 1
            if count == early_stop:
                break
    mat_count, mat_acc = mat_count.cpu().numpy(), mat_acc.cpu().numpy()
    return mask_size_tot, dac_tot, mat_count, mat_acc
"""
------- Test of above code  ----------
"""


from pathlib import Path

import conv_model
import custom_dataset
import torch.nn.functional as F
import zarr
from data_split import StratifiedGroupKFold_custom
from lightning_parallel_training import LightningModelV2
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import BatchSampler, DataLoader
from torchvision.transforms import v2

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")

# select device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define path for real and fake dataset
batch_size = 8
fold = 0
split = "train"
mode = "ref"
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
                                                          org_to_trg_label=None, #[(0, 1), (0, 2), (0, 3)],
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

# visualize attribution map of multiple images when compared to their counterfactuals.

attr_methods = [D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam, Residual_attr, Random_attr]
attr_names = ["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam", "Residual", "Random"]

# plot_attr_img(VGG_model, dataloader_real_fake, fig_directory, name_fig="visualize_attribution", num_img=32,
#               attr_methods=attr_methods, attr_names=attr_names,
#               percentile=98)


# plot_attr_mask_img(VGG_model, dataloader_real_fake, fig_directory, name_fig="visualize_attribution_mask",
#                    num_img=24, steps=200, selected_mask=[0, 50, 100],
#                    attr_methods=attr_methods, attr_names=attr_names,
#                    percentile=98)


attr_methods = [D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam, Residual_attr, Random_attr][:1]
attr_names = ["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam", "Residual", "Random"][:1]
mask_size_tot, dac_tot, mat_count, mat_acc = dac_curve_computation(VGG_model, dataloader_real_fake,
                                                                   attr_methods=attr_methods,
                                                                   attr_names=attr_names,
                                                                   batch_size_mask=512,
                                                                   steps=200, shift=0.7, head_tail=(1, 5),
                                                                   percentile=98,
                                                                   size_closing=10,
                                                                   size_gaussblur=11,
                                                                   early_stop=1, #None,
                                                                   method_kwargs_dict=None, #{"D_GradCam": {"num_layer": -3}}
                                                                   attr_kwargs_dict=None)

tot_acc = mat_acc.sum() / mat_count.sum()
mat_acc_norm = mat_acc / mat_count


# mask_size_tot = {key: np.vstack(value) for key, value in mask_size_tot.items()}
# dac_tot = {key: np.vstack(value) for key, value in dac_tot.items()}
# dac_interp_tot = {key: np.array(list(starmap(lambda x, y: np.interp(mask_size_tot[key].mean(axis=0), x, y), zip(mask_size_tot[key], dac_tot[key]))))
#                   for key in dac_tot.keys()}

# mask_size_df = pd.concat([
#     pd.DataFrame(value).melt(ignore_index=False, value_name="mask_size", var_name="steps").assign(attr_method = np.prod(value.shape) * [key]).reset_index(names="sample")
#     for key, value in mask_size_tot.items()])
# dac_df = pd.concat([
#     pd.DataFrame(value).melt(ignore_index=False, value_name="dac", var_name="steps").assign(attr_method = np.prod(value.shape) * [key]).reset_index(names="sample")
#     for key, value in dac_tot.items()])
# dac_interp_df = pd.concat([
#     pd.DataFrame(value).melt(ignore_index=False, value_name="dac_interp", var_name="steps").assign(attr_method = np.prod(value.shape) * [key]).reset_index(names="sample")
#     for key, value in dac_interp_tot.items()])
# mask_dac_interp_df = pd.merge(pd.merge(mask_size_df,dac_df, on=["sample", "steps", "attr_method"]), dac_interp_df, on=["sample", "steps", "attr_method"])
# mask_dac_interp_df["mask_size_interp"] = mask_dac_interp_df.groupby(["attr_method", "steps"])["mask_size"].transform("mean")


# # mask_dac_df['mask_size_bin'] = pd.cut(mask_dac_df['mask_size'], bins=200)  # Adjust `bins` as needed
# # agg_df = mask_dac_df.groupby(["attr_method", "steps"], observed=True).agg(
# #     mask_size_bin_mean=('mask_size', 'mean')).reset_index()
# # mask_dac_df = pd.merge(mask_dac_df, agg_df, on=["mask_size_bin", "attr_method"], how="left")


# fig, axis = plt.subplots(1, 3, figsize=(15,6))
# axis = axis.flatten()
# # df_dac = pd.DataFrame(data=np.vstack(dac_tot)).melt(ignore_index=False)
# # df.index.name="sample"
# sns.lineplot(data=mask_dac_interp_df, x="steps", y="mask_size", hue="attr_method", ax=axis[0])# , estimator='mean', errorbar=None)
# # sns.lineplot(data=mask_dac_interp_df, x="steps", y="mask_size_interp", hue="attr_method", ax=axis[0])# , estimator='mean', errorbar=None)
# axis[0].set_title('Mask Size Against Steps')
# axis[0].set_xlabel('Steps')
# axis[0].set_ylabel('Mask Size')
# axis[0].grid(True)

# sns.lineplot(data=mask_dac_interp_df, x="mask_size_interp", y="dac_interp", hue="attr_method", ax=axis[1])#, estimator='mean', errorbar=None)
# axis[1].set_title('Mask Size Against DAC')
# axis[1].set_xlabel('mask_size')
# axis[1].set_ylabel('dac')
# axis[1].grid(True)

# sns.heatmap(mat_acc, ax=axis[2], annot=True, fmt=".2f", xticklabels=range(mat_acc.shape[0]), yticklabels=range(mat_acc.shape[0]), vmin=0, vmax=1)
# axis[2].set_title(f'Pred accuracy per class and transition - tot_acc: {tot_acc}')
# axis[2].set_title(f'Acuracy when real and fake well predicted\ntot_acc: {tot_acc:.3f}')
# axis[2].set_xlabel('y_real')
# axis[2].set_ylabel('y_fake')

# plt.savefig(fig_directory / 'mask_size_dac.png', dpi=300, bbox_inches='tight')
