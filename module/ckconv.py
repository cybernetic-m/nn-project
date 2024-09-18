import torch
from torch import nn
import torch.nn.functional as F
from conv_kern_gen import conv_generator
from convKan_kern_gen import convKan_generator


class CKConv(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_dim, omega_0 , dropout_rate, generator_type, data_dim=1, bias = True):

        super(CKConv, self).__init__()

        
        if generator_type == 'conv':
            self.kernel_gen = conv_generator(
            input_channels = data_dim,
            output_channels = input_channels * output_channels,
            hidden_dim = hidden_dim,
            omega_0 = omega_0,
            dropout_rate = dropout_rate,
            bias= bias
            )
        elif generator_type == 'convKan':
            self.kernel_gen = convKan_generator(
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

        self.conv = getattr(F, f"conv{data_dim}d")
        self.sr_change = 1.0

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        self.register_buffer("previous_length", torch.zeros(1).int(), persistent=True)
        

    def forward(self, x):
        # Input shape: torch.Size([1, 2, 100]) Stereo Sample (2 channels)
        rel_pos = self.create_rel_positions(x) # torch.Size([1, 1, 100])
        print("rel_pos:", rel_pos)
        print("rel_pos shape:", rel_pos.shape)
        conv_kernel = self.kernel_gen(rel_pos).view(-1, x.shape[1], x.shape[2])
        print("conv_kernel:", conv_kernel)
        print("conv_kernel shape:", conv_kernel.shape)
        x, conv_kernel = self.causal_padding(x, conv_kernel)
        print("x after causal padding:", x, "\nkernel after causal padding:", conv_kernel)
        print("x shape after causal padding:", x.shape, "\nkernel shape after causal padding:", conv_kernel.shape)
        out = self.conv(x, conv_kernel, bias = self.bias, padding=0)
        print("out shape:", out.shape)
        
        return out
    

    @staticmethod
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
            step_size = previous_length
            n_interval = (previous_length - 1) % sr_change  # n_interval = 9 % 2 = 1 
            length_interval = n_interval*previous_step
            max_rel_pos = 1 - length_interval

        # Case Upsampling: previous_length = 5 (samples), current_length = 10 (samples) => sr_change = 0.5
        else:
            step_size = current_length
            n_interval = (current_length - 1) % (1/sr_change)
            length_interval = n_interval*current_step
            max_rel_pos = 1 + length_interval

        return step_size, max_rel_pos, sr_change


    def create_rel_positions(self, x):

        
        if self.previous_length[0] == 0:
            self.previous_length[0] = x.shape[-1]

        step_size, max_rel_pos, self.sr_change = self.calculate_max(self.previous_length.item(), x.shape[-1])
    
        rel_positions = (
                torch.linspace(-1.0, max_rel_pos, step_size)
                #.cuda()
                .unsqueeze(0)
                .unsqueeze(0)
            )
        return rel_positions

    @staticmethod
    def causal_padding(x, conv_kernel):

        #1. Add zeros to kernel if the kernel_size is even
        if conv_kernel.shape[-1] % 2 == 0:     # Check if it is even
             # Add [1,0] one zero to the left and zero zeros to the right
            conv_kernel = F.pad(conv_kernel, [1,0], value=0.0) # Ex kernel = [1, 1] => kernel = [0, 1, 1]

        #2. Padding of the input: add zeroes to the left of dimensione kernel_size-1 to have causality
        x = F.pad(x, [conv_kernel.shape[-1] - 1,0], value=0.0)   # Ex. x = [1, 2, 3, 4] kernel = [1, 1, 1] => x = [0, 0, 1, 2, 3, 4]

        return x, conv_kernel


tensor = torch.randn(1,2,100)

ckconv = CKConv(
        input_channels = 2,
        output_channels = 2,
        hidden_dim = 32,
        data_dim = 1,
        generator_type= 'conv',
        bias = False,
        omega_0 = 1,
        dropout_rate = 0.5,
    )

out = ckconv(tensor)

print(out)

