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
from preprocessing import invert_audio


class ckaugment_net(nn.Module):
    def __init__(self, input_channels, output_channels, transforms, hidden_dim, hidden_scale, dropout_rate, generator_type, af_type, omega_0, device):
        super(ckaugment_net,self).__init__()

        self.net = nn.ModuleList()

        for transform in transforms:
            self.net.append(transform)
            self.net.append(CKConv(
                input_channels,
                output_channels,
                output_len = 0,
                hidden_dim=hidden_dim,
                omega_0=omega_0,
                dropout_rate=dropout_rate,
                hidden_scale=hidden_scale,
                generator_type=generator_type,
                af_type=af_type,
                bias = True,
                device=device)
            )

    def forward(self, x):
        out = self.net(x)
        return out
