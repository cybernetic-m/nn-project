import torch.nn as nn
import torch
import torch.nn.functional as F

from ctim_net import CTIM_network

class CTIM(nn.Module):

    def __init__(self, kernel_size, dropout_rate, n_temporal_aware_block, n_filter, in_channels, output_len, num_classes, use_kan = False, cont=False, device='cpu'):
        
        super(CTIM,self).__init__()

        self.ctim_net =  CTIM_network(
            kernel_size = kernel_size, 
            dropout_rate = dropout_rate, 
            n_temporal_aware_block = n_temporal_aware_block, 
            n_filter = n_filter, 
            in_channels = in_channels,
            cont = cont,
            output_len = output_len,
            device = device
        )

        if use_kan == True:
            self.classifier = ...
        else:
            self.classifier = nn.Linear(
                in_features = output_len,
                out_features = num_classes,
                bias = True,
                device=device
            )

        self.num_classes = num_classes

    def forward(self, x):
        x1 = self.ctim_net(x)
        x1 = self.classifier(x1)
        out = F.softmax(x1, dim=1)

        return out

if __name__ == '__main__' :

    device='cuda'

    t = torch.rand(1,2,156,device=device)

    ctim = CTIM(
        kernel_size=2, 
        dropout_rate=0.1, 
        n_temporal_aware_block=3, 
        n_filter=64, 
        in_channels=2,
        cont=True,
        output_len=50,
        num_classes = 7,
        use_kan = False,
        device=device
    )

    out = ctim(t)

    print(torch.sum(out))