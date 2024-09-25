# TO DO: add initialization of weight using gaussian
import torch
from torch import nn
import torch.nn.functional as F
from conv_kern_gen import conv_generator
from convKan_kern_gen import convKan_generator


class CKConv(nn.Module):

    def __init__(self, input_channels, output_channels, output_len = 0, hidden_dim=32, omega_0=1, dropout_rate=0.5, generator_type='conv', bias = True, device='cpu'):

        super(CKConv, self).__init__()

        
        if generator_type == 'conv':
            self.kernel_gen = conv_generator(
            input_channels = 1,
            output_channels = input_channels * output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias,
            device=device
            )
        elif generator_type == 'convKan':
            self.kernel_gen = convKan_generator(
            input_channels = input_channels,
            output_channels = output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias,
            device=device
            )
        else:
            print('error in generator type using conv as default')
            self.kernel_gen = conv_generator(
            input_channels = input_channels,
            output_channels = output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias,
            device=device
            )

        self.conv =  F.conv1d
        self.sr_change = 1.0
        self.device = device
        self.output_channels = output_channels
        self.input_channels = input_channels

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_channels)).to(device)
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        self.register_buffer("previous_length", torch.zeros(1).int(), persistent=True)

        # If output_len = 0 => step_size of kernel_gen will be previous or current length
        # Else output_len > 0 => compute a desired (standard) number of features length for Continuous TIM-net
        if output_len >= 0:
            self.output_len = output_len 
        else:
            print("Error: output_len should be a non-negative number!")

        

    def forward(self, x):
        # Input shape: torch.Size([1, 2, 100]) Stereo Sample (2 channels)
        rel_pos = self.create_rel_positions(x) # torch.Size([1, 1, 100])
        #print("rel_pos:", rel_pos)
        #print("rel_pos shape:", rel_pos.shape)
        #print("x_shape", x.shape)
        conv_kernel = self.kernel_gen(rel_pos).view(self.output_channels, self.input_channels, x.shape[2]) # [1, 4, 100] -> [2, 2, 100] [out_ch, in_ch, kern_size]
        #print("conv_kernel:", conv_kernel)
        #print("conv_kernel shape:", conv_kernel.shape)
        x, conv_kernel = self.causal_padding(x, conv_kernel)
        #print("x after causal padding:", x, "\nkernel after causal padding:", conv_kernel)
        #print("x shape after causal padding:", x.shape, "\nkernel shape after causal padding:", conv_kernel.shape)
        out = self.conv(x, conv_kernel, bias = self.bias, padding=0)
        #print("out shape:", out.shape)
        
        return out
    
    def calculate_max(self, previous_length, current_length):

        # Calculate Sampling Rate change
        # Ex. previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2 (double sampling rate)
        # Ex. previous_length = 5 (samples), current_length = 10 (samples) => sr_change = 0.5 (half sampling rate)
        sr_change = previous_length/current_length
        #print("sr_change", sr_change) 

        # Compute Current Step Size
        # Ex. previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2
        # The normalized sequence is in between [-1,1]. The length of the window is 2.0.
        previous_step = 2.0 / (previous_length - 1) # previous_step = 2.0 / 9 (number of interval in between 10 samples) = 0.2
        current_step = previous_step * sr_change # current_step = 0.2 * 2 = 0.4 (it means that I divide [-1,1] in interval of dim 0.4, less samples taken)

        # Compute Maximum Relative Position
        # Case Downsampling: previous_length = 10 (samples), current_length = 5 (samples) => sr_change = 2
        if sr_change > 1:
            n_interval = (previous_length - 1) % sr_change  # n_interval = 9 % 2 = 1 
            length_interval = n_interval*previous_step
            max_rel_pos = 1 - length_interval
            
        # Case Upsampling: previous_length = 5 (samples), current_length = 10 (samples) => sr_change = 0.5
        else:
            n_interval = (current_length - 1) % (1/sr_change)
            length_interval = n_interval*current_step
            max_rel_pos = 1 + length_interval
            
        return max_rel_pos, sr_change


    def create_rel_positions(self, x):

        
        if self.previous_length[0] == 0:
            self.previous_length[0] = x.shape[-1]

        max_rel_pos, self.sr_change = self.calculate_max(self.previous_length.item(), x.shape[-1])
    
        rel_positions = (
                torch.linspace(-1.0, max_rel_pos, x.shape[2])
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
        return rel_positions

    def causal_padding(self, x, conv_kernel):
        
        #1. Add zeros to kernel if the kernel_size is even
        if conv_kernel.shape[-1] % 2 == 0:     # Check if it is even
             # Add [1,0] one zero to the left and zero zeros to the right
            conv_kernel = F.pad(conv_kernel, [1,0], value=0.0) # Ex kernel = [1, 1] => kernel = [0, 1, 1]

        if self.output_len == 0:
            #2. Padding of the input: add zeroes to the left of dimensione kernel_size-1 to have causality
            x = F.pad(x, [conv_kernel.shape[-1] - 1,0], value=0.0)   # Ex. x = [1, 2, 3, 4] kernel = [1, 1, 1] => x = [0, 0, 1, 2, 3, 4]
        else:
            if x.shape[-1] % 2 == 0:
                x = F.pad(x, [self.output_len,0], value=0.0)
            else:
                x = F.pad(x, [self.output_len - 1,0], value=0.0)

        return x, conv_kernel
if __name__ == '__main__' :
    tensor = torch.randn(1,2,145)

    ckconv = CKConv(
            input_channels = 2,
            output_channels = 2,
            hidden_dim = 32,
            generator_type= 'conv',
            output_len=50,
            bias = False,
            omega_0 = 1,
            dropout_rate = 0.5,
        )

    out = ckconv(tensor)

