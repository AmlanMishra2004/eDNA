import torch
import torch.nn as nn
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
#   dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    
class Zurich(nn.Module):
    def __init__(self, num_classes):
        super(Zurich, self).__init__()
        # padding = (1*(60-1) - 60 + 7)/2 = 3
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3)
        self.leakyrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(60*4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
class CNN1(nn.Module):
    def __init__(self, num_classes):
        super(CNN1, self).__init__()
        # padding = (1*(60-1) - 60 + 3)/2 = 1
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
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