import sys
import os
import torch.nn as nn
import torch

# Get the absolute paths of the directories containing the modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../module'))
dataloader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataloader'))
# Add these directories to sys.path
sys.path.append(module_path)
sys.path.append(dataloader_path)

from ckconv import CKConv
from temporal_aware_block import TempAw_Block
from preprocessing import invert_audio


class CTIM_network(nn.Module):

    def __init__(self, kernel_size, dropout_rate, n_temporal_aware_block, n_filter, in_channels, is_siren, omega_0=1, ck=False, device='cpu'):
        
        super(CTIM_network,self).__init__()
        if ck:
            self.conv_forward = CKConv(
                input_channels=in_channels,
                output_channels=n_filter,
                is_siren=is_siren,  
                device=device
            )

            self.conv_reverse = CKConv(
                input_channels=in_channels,
                output_channels=n_filter,
                is_siren=is_siren,
                device=device
            )
        else:
            self.conv_forward = nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filter,
                kernel_size=1,
                device=device
            )

            self.conv_reverse = nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filter,
                kernel_size=1,
                device=device
            )            

        self.TempAw_Blocks_forward = nn.ModuleList()
        self.TempAw_Blocks_reverse = nn.ModuleList()

        for n in range(n_temporal_aware_block):
            dilation_rate = 2**n
            TempAw_Block_n_forward = TempAw_Block(
                n_filter=n_filter,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate,
                ck = ck,
                omega_0=omega_0,
                is_siren=is_siren,
                device=device
            )
            TempAw_Block_n_reverse = TempAw_Block(
                n_filter=n_filter,
                kernel_size=kernel_size,
                dilation_rate= dilation_rate,
                dropout_rate= dropout_rate,
                ck = ck,
                omega_0=omega_0, 
                is_siren=is_siren,  
                device=device
            )
            self.TempAw_Blocks_forward.append(TempAw_Block_n_forward)
            self.TempAw_Blocks_reverse.append(TempAw_Block_n_reverse)  
      
        # Trainable parameters generated by Uniform probability distribution for "Dynamic Fusion"
        self.weights = nn.Parameter(torch.rand(n_temporal_aware_block, 1), requires_grad=True).to(device)  
  
    def forward(self, x):
        reverse_input = invert_audio(x)

        x_forward = self.conv_forward(x)
        x_reverse = self.conv_reverse(reverse_input)

        g_list = []
        batch_size = x.shape[0]
        for tab_forward,tab_reverse in zip(self.TempAw_Blocks_forward, self.TempAw_Blocks_reverse):
            x_forward = tab_forward(x_forward)
            x_reverse = tab_reverse(x_reverse)
            #print()
            x_sum = torch.add(x_forward, x_reverse)
            #print(x_sum.shape)
            g_tensor = torch.mean(x_sum, dim=2)
            #print("g_tensor",g_tensor.shape)
            g_list.append(g_tensor)
        g = torch.cat(g_list).view(batch_size, -1, len(g_list))
        #print("g",g.shape)
        # Dynamic Fusion block
        #print("self.weights",self.weights.shape)
        gdrf = torch.matmul(g, self.weights) # Weighted summation at the end
        #print("gdrf",gdrf.shape)
        out = gdrf.view(batch_size, gdrf.shape[1]) # Transpose 
        #print(out.shape)

        return out
    
if __name__ == '__main__' :
    t = torch.rand(1,2,156)

    ctim = CTIM_network(
        kernel_size=2, 
        dropout_rate=0.1, 
        n_temporal_aware_block=3, 
        n_filter=64, 
        in_channels=2,
        cont=True,
        output_len=50,
    )

    out = ctim(t)
