import torch
import torch.nn as nn

'''
For now, I just stole this architecture from
"Fast processing of environmental DNA metabarcoding 
sequence data using convolutional neural networks"
- it seems somewhat silly, and will likely need to be replaced'''
class My_CNN(nn.Module):
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
    