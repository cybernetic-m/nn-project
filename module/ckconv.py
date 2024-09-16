import torch
from torch import nn
import torch.nn.functional as F
from conv_kern_gen import conv_generator
from kan_kern_gen import kan_generator
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
        self.sr_change = 1.0
        self.bias = bias

        self.register_buffer("previous_length", torch.zeros(1).int(), persistent=True)
        

    def forward(self, x):

        rel_pos = self.create_rel_positions(x)
        conv_kernel = self.kernel_gen(rel_pos).view(-1, x.shape[1], x.shape[2])
        x, conv_kernel = self.causal_padding(x, conv_kernel)
        out = self.conv(x, conv_kernel, bias = self.bias, padding=0)
        
        return out
    
    def calculate_max(previous_length, current_length):

        # Calculate Sampling Rate change
        # Ex. previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2 (double sampling rate)
        # Ex. previous_length = 5 (samples), current_length = 10 (samples) => sr_change = 0.5 (half sampling rate)
        sr_change = previous_length/current_length 

        # Compute Current Step Size
        # Ex. previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2
        # The normalized sequence is in between [-1,1]. The length of the window is 2.0.
        previous_step = 2.0 / (previous_length - 1) # previous_step = 2.0 / 9 (number of interval in between 10 samples) = 0.2
        current_step = previous_step * sr_change # current_step = 0.2 * 2 = 0.4 (it means that I divide [-1,1] in interval of dim 0.4, less samples taken)

        # Compute Maximum Relative Position
        # Case Downsampling: previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2
        if sr_change > 1:
            n_interval = (previous_length - 1) % sr_change  # n_interval = 9 % 2 = 1 (1 )
            length_interval = n_interval*previous_step
            max_rel_pos = 1 - length_interval

        # Case Upsampling: previous_length = 5 (samples), current_length = 10 (samples) => sr_change = 0.5
        else:
            n_interval = (current_length - 1) % (1/sr_change)
            length_interval = n_interval*current_step
            max_rel_pos = 1 + length_interval

        return current_step, max_rel_pos, sr_change


    def create_rel_positions(self, x):

        if self.previous_length[0] == 0:
            self.previous_length[0] = x.shape[-1]

        step_size, max_rel_pos, self.sr_change = self.calculate_max(self.previous_length.item(), current_length=x.shape[-1])

        rel_positions = (
                torch.linspace(-1.0, max_rel_pos, step_size)
                .cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )
        return rel_positions

    def causal_padding(x, conv_kernel):

        #1. Add zeros to kernel if the kernel_size is even
        if conv_kernel.shape[-1] % 2 == 0:     # Check if it is even
             # Add [1,0] one zero to the left and zero zeros to the right
            conv_kernel = F.pad(conv_kernel, [1,0], value=0.0) # Ex kernel = [1, 1] => kernel = [0, 1, 1]

        #2. Padding of the input: add zeroes to the left of dimensione kernel_size-1 to have causality
        x = F.pad(x, [conv_kernel.shape[-1] - 1,0], value=0.0)   # Ex. x = [1, 2, 3, 4] kernel = [1, 1, 1] => x = [0, 0, 1, 2, 3, 4]

        return x, conv_kernel

'''
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
'''