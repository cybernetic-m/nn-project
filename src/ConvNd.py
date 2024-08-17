import torch
from torch import nn

class ConvNd(nn.module):
    def __init__(self, hidden_dim, padding, stride, in_channels, out_channels):
        super.__init__(ConvNd)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

    
    def forward(self, x):
        kernel, data_dim = self.get_kernel(x)
        conv_function = getattr(torch.nn.functional, f"conv{data_dim}d")
        return conv_function(x, kernel, padding=self.padding, stride=self.stride)

    def get_kernel(self, x):
        ...