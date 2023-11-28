import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
#                 dilation=1, groups=1, bias=True, padding_mode='zeros',
#                 device=None, dtype=None)

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
#                 dilation=1, groups=1, bias=True, padding_mode='zeros',
#                 device=None, dtype=None)

same_padding = lambda stride, in_width, out_width, filter_size : math.floor((stride * (out_width - 1) - in_width + filter_size)/2)

class Zurich(nn.Module):
    def __init__(self, stride=1, in_width=60, num_classes=156):
        super(Zurich, self).__init__()
        self.name = "Zurich"
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=same_padding(stride, in_width, in_width, 7))
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
    
class CNN1(nn.Module):
    def __init__(self, num_classes):
        super(CNN1, self).__init__()
        self.name = "CNN1"
        # padding = (1*(60-1) - 60 + 3)/2 = 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
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

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
#                 dilation=1, groups=1, bias=True, padding_mode='zeros',
#                 device=None, dtype=None)

class SmallCNN1(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN1, self).__init__()
        self.name = "SmallCNN1"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
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
        self.conv1 = nn.Conv1d(4, 4, 7, padding=same_padding(stride, in_width, in_width, 3))
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

# Similar to SmallCNN1_1 except with a higher number of output channels and thus
#    a larger number of nodes in the FC layer
class SmallCNN1_2(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN1_2, self).__init__()
        self.name = "SmallCNN1_1"
        self.conv1 = nn.Conv1d(4, 128, 7, padding=same_padding(stride, in_width, in_width, 3))
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
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
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
        self.conv1 = nn.Conv1d(4, 4, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(4, 4, 5, padding=same_padding(stride, in_width, in_width, 5))
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
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
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
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
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
        super(SmallCNN2, self).__init__()
        self.name = "SmallCNN2"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
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

# undeveloped copy of smallcnn2
class SmallCNN2_5(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN2, self).__init__()
        self.name = "SmallCNN2"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
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

# Same as SmallCNN2 except one more layer, pooling 2->3, &-> larger FC layer (1856 -> 1792)
# 79% at lr=0.0005, 0.001 was also good
class SmallCNN3(nn.Module):
    def __init__(self, stride, in_width, num_classes):
        super(SmallCNN3, self).__init__()
        self.name = "SmallCNN3"
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
        self.conv3 = nn.Conv1d(64, 64, 5, padding=same_padding(stride, in_width, in_width, 5))
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
        self.conv1 = nn.Conv1d(4, 32, 3, padding=same_padding(stride, in_width, in_width, 3))
        self.conv2 = nn.Conv1d(32, 64, 8, padding=same_padding(stride, in_width, in_width, 8))
        self.conv3 = nn.Conv1d(64, 64, 5, padding=same_padding(stride, in_width, in_width, 5))
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
    
# model = Zurich(num_classes=5)
# input = torch.randn(3, 4, 60)
# output = model(input)
# print(f"OUTPUT\n{output}\n")


###############################################################################
# End of viable models
###############################################################################

# ABANDONED 
#   ...because their data was formatted as (batch_size, height=150, width=4, channels=1)
#   and they did depthwise-seperable convolutions on that, which is pointless
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
            nn.Conv2d(4, 4, (1,filter_size), stride=stride, padding=padding, groups=4, bias=True), # depthwise
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=True), # pointwise, 4 channels output
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        # Note: groups=in_channels makes it so that each input channel is convolved with its own set of filters, producing one output channel per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # print(f"SHAPE: {x.shape}")
        x = self.relu(self.depthwise(x))
        # print(f"SHAPE: {x.shape}")
        x = self.relu(self.pointwise(x))
        # print(f"SHAPE: {x.shape}")
        return x