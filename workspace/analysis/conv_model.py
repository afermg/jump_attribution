import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple

class VGG(nn.Module):
    def __init__(self, img_depth, img_size, lab_dim, n_conv_block, n_conv_list, n_lin_block):
        super().__init__()
        self.img_depth = img_depth
        self.img_size = img_size
        self.img_size = img_size
        self.lab_dim = lab_dim
        self.n_conv_block = n_conv_block
        self.n_conv_list = n_conv_list
        self.n_lin_block = n_lin_block
        self.fc_dim = 12 * (2 ** (self.n_conv_block - 1)) * ((self.img_size // (2 ** self.n_conv_block)) ** 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.sequence = nn.Sequential(
                        *[self.conv_block((12 * (2 ** (i-1)) if i != 0 else self.img_depth),
                                          (12 * (2 ** i) if i!=0 else 12),
                                          self.n_conv_list[i])
                                     for i in range(self.n_conv_block)],
                        nn.Flatten(),
                        *[self.linear_block(self.fc_dim // (4 ** i), self.fc_dim // (4 ** (i + 1)))
                          for i in range(self.n_lin_block - 1)],
                        nn.Linear(self.fc_dim // (4 ** (self.n_lin_block - 1)), self.lab_dim))
        
    def conv_block(self, in_ch, out_ch, num_conv):
        return nn.Sequential(
            *sum([(nn.Conv2d(in_channels=(in_ch if i==0 else out_ch), out_channels=out_ch, 
                             kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(out_ch),
                   self.relu)
              for i in range(num_conv)], ()),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def linear_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            self.relu,
            self.drop
        )

    def forward(self, x):
        return self.sequence(x)



class VGG_ch(nn.Module):
    def __init__(self, img_depth, img_size, lab_dim, conv_n_ch, n_conv_block, n_conv_list, n_lin_block, p_dropout):
        super().__init__()
        self.img_depth = img_depth
        self.img_size = img_size
        self.img_size = img_size
        self.lab_dim = lab_dim
        self.n_conv_block = n_conv_block
        self.n_conv_list = n_conv_list
        self.n_lin_block = n_lin_block
        self.conv_n_ch = conv_n_ch
        self.fc_dim = self.conv_n_ch * (2 ** (self.n_conv_block - 1)) * ((self.img_size // (2 ** self.n_conv_block)) ** 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=p_dropout)
        self.sequence = nn.Sequential(
                        *[self.conv_block((self.conv_n_ch * (2 ** (i-1)) if i != 0 else self.img_depth),
                                          (self.conv_n_ch * (2 ** i) if i!=0 else self.conv_n_ch),
                                          self.n_conv_list[i])
                                     for i in range(self.n_conv_block)],
                        nn.Flatten(),
                        *[self.linear_block(self.fc_dim // (4 ** i), self.fc_dim // (4 ** (i + 1)))
                          for i in range(self.n_lin_block - 1)],
                        nn.Linear(self.fc_dim // (4 ** (self.n_lin_block - 1)), self.lab_dim))

    def conv_block(self, in_ch, out_ch, num_conv):
        return nn.Sequential(
            *sum([(nn.Conv2d(in_channels=(in_ch if i==0 else out_ch), out_channels=out_ch,
                             kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(out_ch),
                   self.relu)
              for i in range(num_conv)], ()),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def linear_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            self.relu,
            self.drop
        )

    def forward(self, x):
        return self.sequence(x)



'''
------------------- This code has been took from https://github.com/dlmbl/dlmbl-unet/blob/main/src/dlmbl_unet/unet.py  -------------------
'''


class ConvBlock(torch.nn.Module):
    """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Literal["same", "valid"] = "same",
        ndim: Literal[2, 3] = 2,
    ):
        """
        Args:
            in_channels (int): The number of input channels for this conv block.
            out_channels (int): The number of output channels for this conv block.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an NxN or NxNxN
                kernel for ``ndim=2`` and ``ndim=3``, respectively.
            padding (Literal["same", "valid"], optional): The type of padding to
                use. "same" means padding is added to preserve the input dimensions.
                "valid" means no padding is added. Defaults to "same".
            ndim (Literal[2, 3], optional): Number of dimensions for the convolution operation. Use
                2 for 2D convolutions and 3 for 3D convolutions. Defaults to 2.

        Raises:
            ValueError: If unsupported values are used for padding or ndim.
        """
        super().__init__()
        if padding not in ("valid", "same"):
            msg = f"Invalid string value for padding: {padding=}. Options are same or valid."
            raise ValueError(msg)
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        # define layers in conv pass
        self.conv_pass = torch.nn.Sequential(
            convops[ndim](
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            convops[ndim](
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, tuple(convops.values())):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.conv_pass(x)
        return output


class Downsample(torch.nn.Module):
    """Downsample module for U-Net"""

    def __init__(self, downsample_factor: int, ndim: Literal[2, 3] = 2):
        """
        Args:
            downsample_factor (int): Factor by which to downsample featuer maps.
            ndim (Literal[2,3], optional): Number of dimensions for the downsample operation.
                Defaults to 2.

        """

        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.ndim = ndim
        downops = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        self.downsample_factor = downsample_factor

        self.down = downops[ndim](downsample_factor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        output: torch.Tensor = self.down(x)
        return output


class CropAndConcat(torch.nn.Module):
    def crop(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Center-crop x to match spatial dimensions given by y."""
        x_target_size = x.size()[:2] + y.size()[2:]

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(
        self, encoder_output: torch.Tensor, upsample_output: torch.Tensor
    ) -> torch.Tensor:
        encoder_cropped = self.crop(encoder_output, upsample_output)

        return torch.cat([encoder_cropped, upsample_output], dim=1)


class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[torch.nn.Module] = None,
        ndim: Literal[2, 3] = 2,
    ):
        """A convolutional block that applies a torch activation function.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            activation (torch.nn.Module  |  None, optional): An instance of any torch activation
                function (e.g., ``torch.nn.ReLU()``). Defaults to None for no activation after the
                convolution.
            ndim (Literal[2,3], optional): Number of dimensions for the convolution operation.
                Defaults to 2.
        Raises:
            ValueError: If unsupported values is used for ndim.
        """
        super().__init__()
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        convops = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        self.final_conv = convops[ndim](
            in_channels, out_channels, 1, padding=0
        )  # leave this out
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: Optional[torch.nn.Module] = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: Literal["same", "valid"] = "same",
        upsample_mode: str = "nearest",
        ndim: Literal[2, 3] = 2,
    ):
        """A U-Net for 2D or 3D input that expects tensors shaped like:
        ``(batch, channels, height, width)`` or ``(batch, channels, depth, height, width)``,
        respectively.

        Args:
            depth (int): The number of levels in the U-Net. 2 is the smallest that really makes
                sense for the U-Net architecture.
            in_channels (int): The number of input channels in the images.
            out_channels (int, optional): How many channels the output should have. Depends on your
                task. Defaults to 1.
            final_activation (Optional[torch.nn.Module], optional): Activation to use in final
                output block. Depends on your task. Defaults to None.
            num_fmaps (int, optional): Number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (int, optional): Factor by which to multiply the number of feature maps
                between levels. Level ``l`` will have ``num_fmaps*fmap_inc_factor**l`` feature maps.
                Defaults to 2.
            downsample_factor (int, optional): Factor for down- and upsampling of the feature maps
                between levels. Defaults to 2.
            kernel_size (int, optional): Kernel size to use in convolutions on both sides of the
                UNet. Defaults to 3.
            padding (Literal["same", "valid"], optional): The type of padding to
                use. "same" means padding is added to preserve the input dimensions.
                "valid" means no padding is added. Defaults to "same".
            upsample_mode (str, optional): The upsampling mode to pass to ``torch.nn.Upsample``.
                Usually "nearest" or "bilinear". Defaults to "nearest".
            ndim (Literal[2, 3], optional): Number of dimensions for the UNet. Use 2 for 2D-UNet and
                3 for 3D-UNet. Defaults to 2.

        Raises:
            ValueError: If unsupported values are used for padding or ndim.
        """

        super().__init__()
        if padding not in ("valid", "same"):
            msg = f"Invalid string value for padding: {padding=}. Options are same or valid."
            raise ValueError(msg)
        if ndim not in (2, 3):
            msg = f"Invalid number of dimensions: {ndim=}. Options are 2 or 3."
            raise ValueError(msg)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(
                    fmaps_in, fmaps_out, self.kernel_size, self.padding, ndim=ndim
                )
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                    ndim=ndim,
                )
            )

        self.downsample = Downsample(self.downsample_factor, ndim=ndim)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1],
            self.out_channels,
            self.final_activation,
            ndim=ndim,
        )

    def compute_fmaps_encoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        if level == 0:  # Leave out function
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> Tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
        ----
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (
            level
        )  # Leave out function
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):  # leave out center of for loop
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:  # leave out center of for loop
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output
        output: torch.Tensor = self.final_conv(layer_input)
        return output

'''
------------------- Model for img generator -------------------
'''
class ImgGenerator(nn.Module):

    def __init__(self, generator, style_encoder, batchnorm_dim=0):
        super().__init__()
        self.generator = generator
        self.style_encoder = style_encoder
        self.transform = nn.BatchNorm1d(batchnorm_dim) if batchnorm_dim!=0 else nn.Identity()

    def forward(self, x, y):
        """
        x: torch.Tensor
            The source image
        y: torch.Tensor
            The style image
        """
        style = self.transform(self.style_encoder(y))
        # Concatenate the style vector with the input image
        style = style.unsqueeze(-1).unsqueeze(-1)
        style = style.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, style], dim=1)
        return self.generator(x)

'''
------------------- Model for vector input -------------------
'''

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_L, p_dopout_L, batchnorm=True):
        super(SimpleNN, self).__init__()
        self.relu = nn.ReLU()
        self.sequence = nn.Sequential(
            *[self.linear_block((input_size if i == 0 else hidden_layer_L[i-1]),
                                hidden_layer_L[i],
                                p_dopout_L[i],
                                batchnorm)
              for i in range(len(hidden_layer_L))],
            nn.Linear(hidden_layer_L[-1], num_classes),
            nn.Softmax(dim=1))

    def linear_block(self, in_dim, out_dim, p_dropout, batchnorm=True):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity(),
            self.relu,
            nn.Dropout(p_dropout)
        )

    def forward(self, x):
        return self.sequence(x)



class VectorUNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(VectorUNet, self).__init__()
        # Encoder
        self.relu = nn.ReLU()
        self.encoder_layers = nn.ModuleList()
        self.hidden_dims = hidden_dims.copy() #list is modified at some point so we want to avoid inplace modification
        self.output_dim = output_dim
        for h_dim in self.hidden_dims:
            self.encoder_layers.append(nn.Linear(input_dim, h_dim))
            input_dim = h_dim

        # Bottleneck
        self.bottleneck = nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for h_dim in reversed(self.hidden_dims[:-1]):
            self.decoder_layers.append(nn.Linear(self.hidden_dims[-1] * 2, h_dim))
            self.hidden_dims[-1] = h_dim

        # Final Layer
        self.final_layer = nn.Linear(self.hidden_dims[0] * 2, self.output_dim)

    def forward(self, x):
        encodings = []

        # Encoder Forward Pass
        for layer in self.encoder_layers:
            x = self.relu(layer(x))
            encodings.append(x)

        # Bottleneck
        x = self.relu(self.bottleneck(x))

        # Decoder Forward Pass
        for i, layer in enumerate(self.decoder_layers):
            # Skip connection: concatenate encoding from the encoder with the decoder output
            x = torch.cat([x, encodings[-(i + 1)]], dim=1)
            x = self.relu(layer(x))

        # Final Layer
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.final_layer(x)

        return x

class VectorGenerator(nn.Module):

    def __init__(self, generator, style_encoder, batchnorm_dim=0):
        super().__init__()
        self.generator = generator
        self.style_encoder = style_encoder
        self.transform = nn.BatchNorm1d(batchnorm_dim) if batchnorm_dim!=0 else nn.Identity()

    def forward(self, x, y):
        """
        x: torch.Tensor
            The source image
        y: torch.Tensor
            The style image
        """
        style = self.transform(self.style_encoder(y))
        # Concatenate the style vector with the input image
        x = torch.cat([x, style], dim=1)
        return self.generator(x)
