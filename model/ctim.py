import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import datetime
import shutil
from kan import KAN

# Get the absolute paths of the directories containing the modules
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

# Add these directories to sys.path
sys.path.append(model_path)

from ctim_net import CTIM_network

class CTIM(nn.Module):

    def __init__(self, kernel_size, dropout_rate, n_temporal_aware_block, n_filter, in_channels, num_features, num_classes, is_siren, omega_0=1, use_kan = False, ck=False, device='cpu'):
        
        super(CTIM,self).__init__()

        self.ctim_net =  CTIM_network(
            kernel_size = kernel_size, 
            dropout_rate = dropout_rate, 
            n_temporal_aware_block = n_temporal_aware_block, 
            n_filter = n_filter, 
            in_channels = in_channels,
            ck = ck,
            omega_0=omega_0,
            is_siren=is_siren,
            device = device
        )

        # TO DO (At the moment it is not implemented)
        if use_kan == True:
            self.classifier = KAN(
                width=[num_features, num_classes],
                device=device
                )
        else:
            self.classifier = nn.Linear(
                in_features = num_features,
                out_features = num_classes,
                bias = True,
                device=device
            )

        # The number of classes (EMOVO => 7)
        self.num_classes = num_classes
        self.use_kan = use_kan

        # String of the model for saving
        self.parent_dir = ''

        if ck:
            self.model_name = 'ckTIM'
        elif ck and use_kan:
            self.model_name = 'ckTIMkAN'
        elif use_kan:
            self.model_name = 'TIMkAN'
        else:
            self.model_name = 'TIM'

    def forward(self, x):
        x1 = self.ctim_net(x)
        x1 = self.classifier(x1)
        #print(x1.shape)
        out = F.softmax(x1, dim=-1)

        return out
    
    def training_mode(self):
        if self.use_kan == True:
            self.ctim_net.train(True)
        else:
            self.ctim_net.train(True)
            self.classifier.train(True)
    
    def eval_mode(self):
        if self.use_kan == True:
            self.ctim_net.eval()
        else:
            self.ctim_net.eval()
            self.classifier.eval()

    def save(self, path):
        current_datetime = datetime.datetime.now() # Take the actual date and time 
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S") # Format the string in [2024-06-25_14:06:10]
        # Check if there is a precedent model to remove it
        if self.parent_dir != '':
            shutil.rmtree(self.parent_dir) # remove the precedent model
        self.parent_dir = path + '/' + self.model_name + formatted_datetime 
        os.makedirs(self.parent_dir)
        name = self.parent_dir + '/model.pt'  # name: your_path/2024-06-25_14:06:10/model.pt
        torch.save(self.state_dict(), name) # save the model in the precedent path
        print("saved:", name)
        return self.parent_dir
    
    def load(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)
        print("loaded:", path)


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