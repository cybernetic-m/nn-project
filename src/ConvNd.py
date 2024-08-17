import torch
from torch import nn
import torch.nn.functional as F     

class ConvNd(nn.module):
    def __init__(self, hidden_dim, padding, stride, in_channels, out_channels):
        super.__init__(ConvNd)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        self.Gin = nn.Linear(1, hidden_dim) 
        self.Gout = nn.Linear(hidden_dim, 1)

    
    def forward(self, x):
        # get the kernel to convolve with the input
        kernel, data_dim = self.get_kernel(x)
        #get the appropiate convolution operator for the dimension from torch  
        conv_function = getattr(torch.nn.functional, f"conv{data_dim}d")
        #apply and return the convolution
        return conv_function(x, kernel, padding=self.padding, stride=self.stride)

    def get_kernel(self, x):
        shape = x.shape
        kernel = []
        #iterate over every dimension
        for dim in shape:
            kernLin = []
            #iterate over every coordinate
            for Ci in range(dim):
                #get kernel value at that position
                ker_pos = self.get_pos_kern(Ci)
                kernLin.append(ker_pos)
            kernel.append(kernLin)
        return ..., len(shape)

    def get_pos_kern(self,coord):
        x = F.relu(self.Gin(coord))
        return self.Gout(x)