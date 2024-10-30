#!/usr/bin/env python
# coding: utf-8
from itertools import starmap
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_vis
from torch import Tensor
from torch.nn import Module

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from captum.attr import Saliency, IntegratedGradients, DeepLift, GuidedBackprop, LayerGradCam, LayerActivation
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
from captum.attr import visualization as viz

torch.manual_seed(42)
np.random.seed(42)
from multiprocessing import Pool, cpu_count

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


"""
------- Visualisation of attribution function ----------
"""
# Visualisation function
def _cumulative_sum_threshold(
    values: npt.NDArray, percentile: Union[int, float]
) -> float:
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    # pyre-fixme[7]: Expected `float` but got `ndarray[typing.Any, dtype[typing.Any]]`.
    return sorted_vals[threshold_id]

def _normalize_scale(attr: npt.NDArray, scale_factor: float) -> npt.NDArray:
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0.",
            stacklevel=2,
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def visualize_attribution(attributions, X_real, X_fake, y_real, y_fake, y_hat_real, y_hat_fake,  method_names=None,
                          fig=None, axes=None, percentile=98):
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
        # Ensure attribution is in channel-last format
        attribution = np.transpose(attribution, (1, 2, 0))
        attribution = np.sum(attribution, axis=-1)
        # Plot attribution map with heatmap from -1 to 1
        threshold = _cumulative_sum_threshold(
            np.abs(attribution), percentile
        )
        attribution = _normalize_scale(attribution, threshold)
        im = axes[i + 2].imshow(attribution, cmap="bwr_r", vmin=-1, vmax=1)
        title = method_names[i] if method_names else f"Attribution {i+1}"
        axes[i + 2].set_title(title)
        axes[i + 2].axis("off")

    # Add color bar for the attribution maps
    cbar = fig.colorbar(im, ax=axes[-1], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label("Attribution Score")

    # plt.tight_layout()
    return fig, axes

def plot_attr_img(model, dataloader_real_fake, fig_directory, name_fig, num_img=24,
                  attr_methods=[D_InGrad, IntegratedGradients, DeepLift, D_GuidedGradCam, D_GradCam],
                  attr_names=["D_InputXGrad", "IntegratedGradients", "DeepLift", "D_GuidedGradcam", "D_GradCam"],
                  percentile=98):

    curr_img = 0
    fig, axis = plt.subplots(num_img, len(attr_methods) + 2, figsize=(5 * (len(attr_methods) + 2), 5 * num_img))
    for X_real, X_fake, y_real, y_fake in dataloader_real_fake:
        X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
        X_real.requires_grad, X_fake.requires_grad = True, True
        with torch.no_grad():
            y_hat_real = F.softmax(model(X_real), dim=1).argmax(dim=1)
            y_hat_fake = F.softmax(model(X_fake), dim=1).argmax(dim=1)
        attributions = []
        for attr_method in attr_methods:
            attributions.append(Attribution(model, attr_method, X_fake, X_real, y_fake).sum(dim=1, keepdim=True).detach().cpu().numpy()) # method_kwargs={"num_layer": -3})
            # add method_kwargs, or attr_kwargs depending on the method.
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

"""
------- Mask creation and DAC score --------
"""

def slice_array(indices, steps=1000):
    total_indices = len(indices)
    # Calculate the proportional so that the tail gather more element that the head
    proportions = np.array([100 if i < 8*steps/10 else 500 for i in range(1, steps + 1) ])
    proportions = proportions / proportions.sum()

    # Compute cumulative indices to use for slicing
    cumulative_sizes = (proportions * total_indices).cumsum().astype(int)

    # Fix rounding issues by adjusting the last index
    cumulative_sizes[-1] = total_indices

    # Use slicing based on cumulative indices
    split_arrays = [indices[cumulative_sizes[i-1] if i > 0 else 0 : cumulative_sizes[i]] for i in range(steps)]
    return split_arrays

# def mask_size_growth2(attribution, struc=10, steps=1000):
#     if len(attribution.shape) >2:
#         if len(attribution.shape) > 3:
#             raise Exception("This function only work with single attribution map.")
#         else:
#             attribution = attribution.sum(axis=0)
#     attribution = attribution / np.abs(attribution).max()
#     indices_sort = np.dstack(np.unravel_index(np.argsort(attribution.ravel()), (448, 448))).squeeze()[::-1]
#     mask_sort = np.zeros((448, 448), dtype=np.uint8)
#     sliced_indices = slice_indices(indices_sort, steps=steps)
#     # sliced_indices = np.array_split(indices_sort, 1000)
#     mask_size_tot = []
#     for _slice in sliced_indices:
#         mask_sort[tuple(_slice.T)] = 1
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
#         mask_true = cv2.morphologyEx(mask_sort, cv2.MORPH_CLOSE, kernel)
#         mask_size = np.sum(mask_true)
#         mask_size_tot.append(mask_size)
#     return mask_size_tot

def get_mask(model, attribution, X_real, X_fake, y_real, y_fake, sigma=11, struc=10):
    """
    attribution should be normalized between -1 and 1
    """
    if len(attribution.shape) >2:
        if len(attribution.shape) > 3:
            raise Exception("This function only work with single attribution map.")
        else:
            attribution = attribution.sum(axis=0)
    attribution = attribution / np.abs(attribution).max()
    indices_sort = np.dstack(np.unravel_index(np.argsort(attribution.ravel()), (448, 448))).squeeze()[::-1]
    sliced_indices = slice_array(indices_sort, steps=steps)
    mask_sort = np.zeros((448, 448), dtype=np.uint8)
    for _slice in sliced_indices:
        mask_sort[tuple(_slice.T)] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_size = np.sum(mask)
        mask_weight = cv2.GaussianBlur(mask.astype(np.float), (sigma,sigma),0)

        X_real_weight = mask_weight * X_real
        X_fake_weight = mask_weight * X_fake
        X_diff_rf = X_real_weight - X_fake_weight
        X_hybrid = X_fake + X_diff_rf

        X_fake_norm = normalize_image(copy.deepcopy(X_fake))
        out_fake = run_inference(model, X_fake_norm)

        X_real_norm = normalize_image(copy.deepcopy(X_real))
        out_real = run_inference(model, X_real_norm)

        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        out_copyto = run_inference(model, im_copied_norm)

        imgs = [attribution, X_real_norm, X_fake_norm, im_copied_norm, normalize_image(copied_canvas),
                normalize_image(copied_canvas_to), normalize_image(diff_copied), mask_weight]

        imgs_all.append(imgs)

        mrf_score = out_copyto[0][y_real] - out_fake[0][real_class]
        result_dict[thr] = [float(mrf_score.detach().cpu().numpy()), mask_size]

    return result_dict, img_names, imgs_all



"""
------- Test of above code  ----------
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

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")

# select device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# define path for real and fake dataset
batch_size = 8
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
# plot_attr_img(VGG_model, dataloader_real_fake, fig_directory, name_fiVg="visualize_attribution", num_img=32,
#               attr_methods=attr_methods, attr_names=attr_names,
#               percentile=98)



num_img = 1000
curr_img = 0
attr_tot = None
for X_real, X_fake, y_real, y_fake in dataloader_real_fake:
    X_real, y_real, X_fake, y_fake = X_real.to(device), y_real.to(device), X_fake.to(device), y_fake.to(device)
    X_real.requires_grad, X_fake.requires_grad = True, True
    # with torch.no_grad():
    #     y_hat_real = F.softmax(model(X_real), dim=1).argmax(dim=1)
    #     y_hat_fake = F.softmax(model(X_fake), dim=1).argmax(dim=1)
    attr_batch = Attribution(VGG_model, DeepLift , X_fake, X_real, y_fake).sum(dim=1, keepdim=True).detach().cpu().numpy() # method_kwargs={"num_layer": -3})
    # add method_kwargs, or attr_kwargs depending on the method.
    torch.cuda.empty_cache()
    if attr_tot is None:
        attr_tot = attr_batch
    else:
        attr_tot = np.vstack([attr_tot, attr_batch])
    if len(attr_tot) >= num_img:
        break





# import time
# import seaborn as sns

# start_time = time.perf_counter()
# with Pool(26) as pool: # with Pool(cpu_count()) as pool:
#     processed_batch2 = pool.map(mask_size_growth2, attr_tot)
#     processed_batch2 = np.array(processed_batch2)
# end_time = time.perf_counter()
# print(f"Total execution time: {end_time - start_time} seconds")


# import pandas as pd
# fig, ax = plt.subplots(figsize=(10, 6))
# df = pd.DataFrame(data=processed_batch2).melt(ignore_index=False)
# df.index.name="sample"
# sns.lineplot(data=df, x="variable", y="value", ax=ax)# , estimator='mean', errorbar=None)
# ax.set_title('Mask Size Against Steps')
# ax.set_xlabel('Steps')
# ax.set_ylabel('Mask Size')
# ax.set_xticks(np.linspace(0, 0.2, num=11))  # Optional: Set x-ticks from -1 to 1
# ax.grid(True)  # Optional: Add grid lines for better readability
# plt.savefig(fig_directory / 'mask_size_results2.png', dpi=300, bbox_inches='tight')


sigma = 11
cv2.GaussianBlur(np.random.randint(0, 2, size=(10,10)).astype(np.uint8), (sigma,sigma),0)
