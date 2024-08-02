# (c) 2023 Sam Waggoner
# License: AGPLv3

"""Defines PyTorch models (CNN and linear) intended to be used for classifying
species based on eDNA sequences.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


same_padding = lambda stride, in_width, out_width, filter_size: (
    math.floor(
        (stride * (out_width - 1) - in_width + filter_size) / 2
    )
)

class Zurich(nn.Module):
    def __init__(self, stride=1, in_width=60, num_classes=156):
        super(Zurich, self).__init__()
        self.name = "Zurich"
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7,
                               padding=same_padding(stride, in_width,
                                                    in_width, 7))
        self.leakyrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(60*4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.leakyrelu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Jon_CNN(nn.Module):
    """Written by Jon Donnely. Architecture taken from "Fast processing of
    environmental DNA metabarcoding sequence data using convolutional neural
    networks.
    """
    def __init__(self, in_channels=4, 
                 conv1_out_channels=16, conv1_kernel_size=7,
                 conv2_out_channels=32, conv2_kernel_size=7,
                 conv3_out_channels=64, conv3_kernel_size=7,
                 in_len=150, n_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.in_len = in_len

        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.conv3_out_channels = conv3_out_channels

        self.conv1_kernel_size = conv1_kernel_size
        self.conv2_kernel_size = conv2_kernel_size
        self.conv3_kernel_size = conv3_kernel_size

        print("conv1_out_channels: ", conv1_out_channels)
        print("conv2_out_channels: ", conv2_out_channels)
        print("conv3_out_channels: ", conv3_out_channels)

        # input shape: (batch_size, channels=4, width=sequence_length)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                    out_channels=conv1_out_channels, 
                    kernel_size=conv1_kernel_size,
                    padding=conv1_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_out_channels, 
                    out_channels=conv2_out_channels, 
                    kernel_size=conv2_kernel_size,
                    padding=conv2_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv2_out_channels, 
                    out_channels=conv3_out_channels, 
                    kernel_size=conv3_kernel_size,
                    padding=conv3_kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(in_channels=conv3_out_channels, 
                    out_channels=conv3_out_channels*2, 
                    kernel_size=conv1_kernel_size,
                    padding=conv1_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv3_out_channels*2, 
                    out_channels=conv3_out_channels*4, 
                    kernel_size=conv2_kernel_size,
                    padding=conv2_kernel_size // 2),    
            nn.ReLU(),
            nn.Conv1d(in_channels=conv3_out_channels*4, 
                    out_channels=conv3_out_channels*8, 
                    kernel_size=conv3_kernel_size,
                    padding=conv3_kernel_size // 2),
            nn.ReLU()
        )
        self.last_layer = nn.Linear(in_len//2*conv3_out_channels*8, n_classes)

    def forward(self, x):
        out = self.conv_layers(x)
        # Reshape out to feed into linear layer
        out = out.view(out.shape[0], -1)
        out = self.last_layer(out)
        return out

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
class VariableCNN(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, stride, padding,
                 dropout_channels, pool_kernel_size, activation, input_length, num_classes):
        self.name = "VariableCNN"
        super(VariableCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(len(in_channels)):
            self.conv_layers.append(
                nn.Conv1d(in_channels[i], out_channels[i], conv_kernel_size[i],
                          stride[i], padding[i])
            )
            self.conv_layers.append(activation())
            # if pool_kernel_size[i] > 0:
            #     self.conv_layers.append(
            #         nn.MaxPool1d(pool_kernel_size[i])
            #)
            self.conv_layers.append(
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.conv_layers.append(
                nn.Dropout(dropout_channels[i])
            )
        # Calculate required number of input nodes in dense layer
        for i in range(len(out_channels)):
            input_length = utils.conv1d_output_size(input_length, conv_kernel_size[i], padding[i], stride[i]) / 2
            if input_length <= 0:
                raise ValueError("The given combination of kernel sizes, paddings,"
                                " and stride parameters do not result in a "
                                "positive input length.")
            if pool_kernel_size[i] > 0:
                input_length = input_length // pool_kernel_size[i]
                if input_length <= 0:
                    raise ValueError("The pooling operation does not result in a "
                                    "positive input length.")
        num_nodes = input_length * out_channels[-1]
        if num_nodes <= 0:
            raise ValueError("The given combination of kernel sizes, paddings,"
                            " and stride parameters do not result in a "
                            "positive number of nodes in the linear layer.")
        self.linear_layer = nn.Linear(num_nodes, num_classes)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        # softmax is not applied here because crossentropyloss applies it for us
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# optimized for in_width=60, num_classes=156
# made w/ the same var. names s.t. you can load in the saved weights
# 14 epochs, 0.002 lr, 16 batch size, no oversampling
class Best_12_31(nn.Module):
    def __init__(self):
        super(Best_12_31, self).__init__()
        self.name = "Best_12_31"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=5,
                            stride=1, padding=2),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.MaxPool1d(2))
        self.conv_layers.append(nn.Dropout(0.4))
        self.conv_layers.append(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=11,
                            stride=1, padding=2),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.MaxPool1d(3))
        self.conv_layers.append(nn.Dropout(0.6))

        self.linear_layer = nn.Linear(2048, 156) # works for 64 and 60
        # self.linear_layer = nn.Linear(2304, 156) # for 71

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        # x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        # x = self.softmax(self.linear_layer(x))
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# seq_len=60, batch=64, lr=0.005, epochs=15, oversample=True
class Large_Best(nn.Module):
    def __init__(self):
        super(Large_Best, self).__init__()
        self.name = "Large_Best"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=612, kernel_size=6,
                            stride=1, padding=2), # used to be padding=0
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.2))
        self.conv_layers.append(
            nn.Conv1d(in_channels=612, out_channels=512, kernel_size=6,
                            stride=1, padding=3), # used to be padding=0
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2)) # used to be k=3, s=1. but now it has to halve the length
        self.conv_layers.append(nn.Dropout(0.1))

        self.linear_layer = nn.Linear(8192, 156) # original: 8192 modified: 16384

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # UNCOMMENT these last two lines for evaluate_model.py, and
        # COMMENT these last two lines for main.py!
        # x = x.view(x.size(0), -1)
        # x = self.linear_layer(x)

        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# seq_len=71, batch=64, lr=0.002, epochs=18, oversample=True
class Small_Best(nn.Module):
    def __init__(self):
        super(Small_Best, self).__init__()
        self.name = "Small_Best"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=512, kernel_size=8,
                            stride=1, padding=0),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.5))
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2)) # has to halve the length
        # self.conv_layers.append(nn.MaxPool1d(3))

        self.linear_layer = nn.Linear(16384, 156) # 10752 for kernel of 3

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        # also divide by normalization in the latent space (don't change the input vector)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # print(f"INTERMEDIATE SHAPE: {x.shape}")
        # UNCOMMENT this line for evaluate_model.py, and
        # COMMENT this line for main.py!
        x = x.view(x.size(0), -1)
        # print(f"FLATTENED SHAPE: {x.shape}")
        x = self.linear_layer(x)

        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Small_Best_Updated(nn.Module):
    def __init__(self):
        super(Small_Best_Updated, self).__init__()
        self.name = "Small_Best_Updated"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.5))
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.linear_layer = nn.Linear(17920, 156)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # print(f"Shape after conv layers: {x.shape}")
        # UNCOMMENT these lines for evaluate_model.py, and
        # COMMENT these lines for main.py!
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Small_Best_Updated_2layer(nn.Module):
    def __init__(self):
        super(Small_Best_Updated_2layer, self).__init__()
        self.name = "Small_Best_Updated_2layer"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.5))
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.linear_layer = nn.Linear(17920, 156)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # print(f"Shape after conv layers: {x.shape}")
        # UNCOMMENT these lines for evaluate_model.py, and
        # COMMENT these lines for main.py!
        # x = x.view(x.size(0), -1)
        # x = self.linear_layer(x)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Small_Best_Updated_3layer(nn.Module):
    def __init__(self):
        super(Small_Best_Updated_3layer, self).__init__()
        self.name = "Small_Best_Updated_3layer"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.5))
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.linear_layer = nn.Linear(17920, 156)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # print(f"Shape after conv layers: {x.shape}")
        # UNCOMMENT these lines for evaluate_model.py, and
        # COMMENT these lines for main.py!
        # x = x.view(x.size(0), -1)
        # x = self.linear_layer(x)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Small_Best_Updated_4layer(nn.Module):
    def __init__(self):
        super(Small_Best_Updated_4layer, self).__init__()
        self.name = "Small_Best_Updated_4layer"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=4, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7,
                            stride=1, padding=3),
        )
        self.conv_layers.append(nn.LeakyReLU())
        self.conv_layers.append(nn.Dropout(0.5))
        self.conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.linear_layer = nn.Linear(17920, 156)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # print(f"Shape after conv layers: {x.shape}")
        # UNCOMMENT these lines for evaluate_model.py, and
        # COMMENT these lines for main.py!
        # x = x.view(x.size(0), -1)
        # x = self.linear_layer(x)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class CNN1(nn.Module):
    def __init__(self, num_classes):
        super(CNN1, self).__init__()
        self.name = "CNN1"
        # padding = (1*(60-1) - 60 + 3)/2 = 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(60*128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.softmax(self.fc(x))
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class SmallCNN1(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN1, self).__init__()
        self.name = "SmallCNN1"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                              in_width, 3))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*in_width//2, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Similar to Zurich, except with one dense layer
class SmallCNN1_1(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN1_1, self).__init__()
        self.name = "SmallCNN1_1"
        self.conv1 = nn.Conv1d(4, 4, 7, padding=same_padding(stride, in_width,
                                                             in_width, 3))
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(224, num_classes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Similar to SmallCNN1_1 except with a higher number of output channels and 
#    thus a larger number of nodes in the FC layer
class SmallCNN1_2(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN1_2, self).__init__()
        self.name = "SmallCNN1_1"
        self.conv1 = nn.Conv1d(4, 128, 7, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 3))
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(7168, num_classes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class SmallCNN2(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2, self).__init__()
        self.name = "SmallCNN2"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride,in_width,
                                                              in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.leakyrelu = nn.LeakyReLU() # was previously relu I believe
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856, num_classes) # 64*in_width

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Same as SmallCNN2 except much smaller depth &-> smaller FC layer
class SmallCNN2_1(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2_1, self).__init__()
        self.name = "SmallCNN2_1"
        self.conv1 = nn.Conv1d(4, 4, 3, padding=same_padding(stride, in_width,
                                                             in_width, 3))
        self.conv2 = nn.Conv1d(4, 4, 5, padding=same_padding(stride, in_width, 
                                                             in_width, 5))
        self.leakyrelu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4*60//2, num_classes) # 64*in_width

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# same as SmallCNN2 except small dropout is applied to the first conv layer
class SmallCNN2_2(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2_2, self).__init__()
        self.name = "SmallCNN2_2"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.leakyrelu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.smalldropout = nn.Dropout(0.2)
        self.bigdropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856, num_classes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.smalldropout(x)
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.bigdropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Same as SmallCNN2 except max pool 2->3
class SmallCNN2_3(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2_3, self).__init__()
        self.name = "SmallCNN2_3"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.leakyrelu = nn.LeakyReLU() # was previously relu I believe
        self.pool = nn.MaxPool1d(3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1216, num_classes) # 64*in_width

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# undeveloped copy of smallcnn2
class SmallCNN2_4(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2_4, self).__init__()
        self.name = "SmallCNN2_4"
        self.conv1 = nn.Conv1d(4, 64, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(64, 128, 8, padding=same_padding(stride,
                                                                in_width,
                                                                in_width, 8))
        self.leakyrelu = nn.LeakyReLU() # was previously relu I believe
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856*2, num_classes) # 64*in_width

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# undeveloped copy of smallcnn2
class SmallCNN2_5(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2, self).__init__()
        self.name = "SmallCNN2"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.leakyrelu = nn.LeakyReLU() # was previously relu I believe
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856, num_classes) # 64*in_width

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# chosen by AutoKeras after 63 trials
class SmallCNN2_6(nn.Module):
    def __init__(self, in_width, num_classes):
        super(SmallCNN2_6, self).__init__()
        self.name = "SmallCNN2_6"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(1, in_width,
                                                              in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 3, padding=same_padding(1, in_width,
                                                               in_width, 3))
        self.pool1d = nn.MaxPool1d(2, stride=2, padding=1)
        # ak says stride should be 2 for the pool2d
        self.pool2d = nn.MaxPool2d((2,2), stride=1, padding=1) 
        # for stride 1, number of nodes should be: (1921 for 1d, 1023 for 2d)
        self.fc1 = nn.Linear(3965, num_classes) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        # x = torch.flatten(x, start_dim=1)
        # x = self.pool1d(x)
        x = self.pool2d(x)
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.dropout2(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Same as SmallCNN2 except one more layer, pooling 2->3, &-> larger FC layer
#   (1856 -> 1792)
# 79% at lr=0.0005, 0.001 was also good
class SmallCNN3(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN3, self).__init__()
        self.name = "SmallCNN3"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.conv3 = nn.Conv1d(64, 64, 5, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 5))
        self.leakyrelu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856, num_classes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# Same as SmallCNN3, except added 0.2 dropout on the conv layers
class SmallCNN3_1(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN3_1, self).__init__()
        self.name = "SmallCNN3_1"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width,
                                                               in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 8))
        self.conv3 = nn.Conv1d(64, 64, 5, padding=same_padding(stride,
                                                               in_width,
                                                               in_width, 5))
        self.leakyrelu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1856, num_classes)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.dropout1(x)
        x = self.leakyrelu(self.conv2(x))
        x = self.dropout1(x)
        x = self.leakyrelu(self.conv3(x))
        x = self.dropout1(x)
        x = self.pool(x)
        x = self.dropout2(x)
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Linear1(nn.Module):
    def __init__(self, in_width, num_classes):
        super(Linear1, self).__init__()
        self.name = "Linear1"
        self.fc1 = nn.Linear(4*in_width, 320)
        self.fc2 = nn.Linear(320, 400)
        self.fc3 = nn.Linear(400, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.relu(self.fc3(x))
        x = self.dropout2(x)
        x = F.softmax(x, dim=1)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class Linear2(nn.Module):
    def __init__(self, in_width, num_classes):
        super(Linear2, self).__init__()
        self.name = "Linear1"
        self.fc1 = nn.Linear(4*in_width, 320)
        self.fc2 = nn.Linear(320, num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # flatten all dims except for batch
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()




























###############################################################################
# End of viable models
###############################################################################

# ABANDONED 
#   ...because their data was formatted as (batch_size, height=150, width=4,
#   channels=1).
#   They did depthwise-seperable convolutions on that, which is pointless
#   because the depth is 1 (channels) so it is equivalent to a 1D convolution.
class ZurichModel(nn.Module):
    # assumes (pytorch) input size of (bs, channels=4, height=1, width=60)
    def __init__(self):
        super(ZurichModel, self).__init__()

        stride = 1
        input_width = 60
        output_width = 60
        filter_size = 7
        padding = (stride * (output_width - 1) - input_width + filter_size)/2

        self.conv_layers = nn.Sequential(
            # depthwise
            nn.Conv2d(4, 4, (1,filter_size), stride=stride, padding=padding,
                      groups=4, bias=True),
            # pointwise
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(216, 128) # Input features,  Output features
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 156) 
        

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.dense1(x)
        # add leaky relu and 
        x = self.dense2(x)        
        pred = self.dense3(x)
        return pred

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


# ABANDONED
# ...because of the reasons explained above
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        # Note: groups=in_channels makes it so that each input channel is
        # convolved with its own set of filters, producing one output channel
        # per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # print(f"SHAPE: {x.shape}")
        x = self.relu(self.depthwise(x))
        # print(f"SHAPE: {x.shape}")
        x = self.relu(self.pointwise(x))
        # print(f"SHAPE: {x.shape}")
        return x