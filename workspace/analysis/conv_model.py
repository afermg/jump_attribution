import torch
import torch.nn as nn

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
