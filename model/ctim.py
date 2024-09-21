import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import datetime

# Get the absolute paths of the directories containing the modules
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

# Add these directories to sys.path
sys.path.append(model_path)

from ctim_net import CTIM_network

class CTIM(nn.Module):

    def __init__(self, kernel_size, dropout_rate, n_temporal_aware_block, n_filter, in_channels, output_len, num_classes, use_kan = False, tab_cont=False, device='cpu'):
        
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

        if tab_cont and use_kan:
            self.model_name = 'ctimKan'
        elif tab_cont:
            self.model_name = 'ctim'
        elif use_kan:
            self.model_name = 'timKan'
        else:
            self.model_name = 'tim'

    def forward(self, x):
        x1 = self.ctim_net(x)
        x1 = self.classifier(x1)
        out = F.softmax(x1, dim=1)

        return out
    
    def training_mode(self):
        self.ctim_net.train()
    
    def eval_mode(self):
        self.ctim_net.eval()

    def save(self):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        name = self.model_name + formatted_datetime + '.pt'
        torch.save(self.state_dict(), name)
        print("saved:", name)
    
    def load(self, formatted_datetime):
        name = self.model_name + formatted_datetime + '.pt'
        self.load_state_dict(torch.load(name))
        print("loaded:", name)


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