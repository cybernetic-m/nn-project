from TemporalAwareBlock import TempAw_Block
from preprocessing import invert_audio
#from DynFus import Dynamic_Fusion

import torch.nn as nn
import torch
import torch.nn.functional as F

class CTIM_network(nn.Module):

    def __init__(self, kernel_size, dropout_rate, n_temporal_aware_block, n_filter, in_channels):
        
        super(CTIM_network,self).__init__()
        
        self.conv_forward = nn.Conv1d(
            in_channels= in_channels,
            out_channels= n_filter,
            kernel_size=1,
            dilation = 1
        )

        self.conv_reverse = nn.Conv1d(
            in_channels= in_channels,
            out_channels= n_filter,
            kernel_size=1,
            dilation = 1
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
            )
            TempAw_Block_n_reverse = TempAw_Block(
                n_filter=n_filter,
                kernel_size=kernel_size,
                dilation_rate= dilation_rate,
                dropout_rate= dropout_rate,
            )
            self.TempAw_Blocks_forward.append(TempAw_Block_n_forward)
            self.TempAw_Blocks_reverse.append(TempAw_Block_n_reverse)  
        
        self.glob_avd_1d = nn.AvgPool1d(
            kernel_size=kernel_size, 
            stride=1
        )
  
    def forward(self, x):
        reverse_input = invert_audio(x)

        x_forward = self.conv_forward(x)
        x_reverse = self.conv_reverse(reverse_input)

        g_list = []
        for tab_forward,tab_reverse in zip(self.TempAw_Blocks_forward, self.TempAw_Blocks_reverse):
            x_forward = tab_forward(x_forward)
            x_reverse = tab_reverse(x_reverse)
            x_sum = torch.add(x_forward, x_reverse)
            g_tensor = self.glob_avd_1d(x_sum)
            g_list.append(g_tensor)
        return g_list

t = torch.rand(1,2,10)

ctim = CTIM_network(
    kernel_size=2, 
    dropout_rate=0.1, 
    n_temporal_aware_block=3, 
    n_filter=64, 
    in_channels=2
)




