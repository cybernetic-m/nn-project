import torch
from torch import nn
import torch.nn.functional as F
from conv_kern_gen import conv_generator
from mlp_kern_gen import mlp_generator

class ckconv(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_dim, hidden_dim, omega_0 , dropout_rate, generator_type, data_dim=1, bias = True):

        super(ckconv, self).__init__()
        if generator_type == 'mlp':
            self.kernel_gen = mlp_generator(
            input_channels = kernel_dim,
            output_channels = output_channels,
            hidden_dim = hidden_dim,
            kernel_dim=kernel_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias
            )
        elif generator_type == 'conv':
            self.kernel_gen = conv_generator(
            input_channels = input_channels,
            output_channels = output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias
            )
        else:
            print('error in generator type using conv as default')
            self.kernel_gen = conv_generator(
            input_channels = input_channels,
            output_channels = output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias
            )
        self.kernel_dim = kernel_dim
        self.conv = getattr(F, f"conv{data_dim}d")

    def forward(self, x):
        x_shape = x.shape
        
        rel_pos = self.create_rel_positions(x)
        
        conv_kernel = self.kernel_gen(rel_pos).view(-1, x_shape[1], *x_shape[2:])
        
        out = self.conv(x, conv_kernel)
        
        return out

    def create_rel_positions(self, x):
        rel_positions = (
                torch.linspace(-1.0, x.shape[-1], self.kernel_dim)
                .cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )
        return rel_positions
    
kernel_dim = 20
    
tensor = torch.randn(1,2,100, device='cuda')

conv = ckconv(
        input_channels = kernel_dim,
        output_channels = 2,
        hidden_dim = 32,
        kernel_dim=kernel_dim,
        omega_0 = 1,
        dropout_rate = 0.5,
        generator_type='mlp',
        bias= False
).cuda()

out = conv(tensor)

print(out)
print(out.shape)