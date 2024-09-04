import torch
from torch import nn
import torch.nn.functional as F     

class ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, data_dim, kernel_size, dilation=1, padding=0, stride=1, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_dim = data_dim  
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.Gin = nn.Linear(self.kernel_size.shape[0], hidden_dim) 
        self.Gout = nn.Linear(hidden_dim, self.kernel_size.shape[0])

    
    def forward(self, x):
        # get the kernel to convolve with the input
        kernel, data_dim = self.get_kernel()       
        #get the appropiate convolution operator for the dimension from torch
        print(f"conv{data_dim}d")
        conv_function = getattr(torch.nn.functional, f"conv{data_dim}d")
        #apply and return the convolution
        return conv_function(x, kernel, padding=self.padding, dilation=self.dilation, stride=self.stride)

    def get_kernel(self):
        shape = self.kernel_size.shape
        kernel = []
        #iterate over every dimension
        for dim in shape:
            # Create tensor of coordinates from 0 to dim-1
            coords = torch.arange(dim, dtype=torch.float).unsqueeze(0)
            
            # Apply the function to each position in the coordinates
            kernLin = []
            for Ci in coords:
                print(Ci)
                kernLin.append(self.get_pos_kern(Ci))
            
            # Stack the results into a tensor
            kernel.append(torch.stack(kernLin))

        # Stack all kernel tensors along a new dimension
        kernel = torch.stack(kernel)
        print(shape)
        print(self.kernel_size)
        print(kernel)
        return kernel, len(shape)

    def get_pos_kern(self,coord):
        x = F.relu(self.Gin(coord))
        return self.Gout(x)
    
layer = ConvNd(3, 3 , kernel_size=torch.tensor([2.,1.,1.]))
input_tensor = torch.randn(32, 3, 10000)
layer(input_tensor)