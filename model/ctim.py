import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import datetime
import shutil

# Get the absolute paths of the directories containing the modules
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model'))

# Add these directories to sys.path
sys.path.append(model_path)

from ctim_net import CTIM_network

class CTIM(nn.Module):

    def __init__(self,
                kernel_size,
                dropout_rate,
                n_temporal_aware_block,
                n_filter, in_channels,
                num_features,
                num_classes,
                af_type='sin',
                omega_0=1,
                generator_type='conv',
                hidden_scale=1,
                ck=False,
                device='cpu'):
        
        super(CTIM,self).__init__()

        self.ctim_net =  CTIM_network(
            kernel_size = kernel_size, 
            dropout_rate = dropout_rate, 
            n_temporal_aware_block = n_temporal_aware_block, 
            n_filter = n_filter, 
            in_channels = in_channels,
            ck = ck,
            omega_0=omega_0,
            af_type=af_type,
            generator_type = generator_type,
            hidden_scale=hidden_scale,
            device = device
        ).to(device)

    
        self.classifier = nn.Linear(
            in_features = num_features,
            out_features = num_classes,
            bias = True,
            device=device
        )

        # The number of classes (EMOVO => 7)
        self.num_classes = num_classes
        self.generator_kan = generator_type

        # String of the model for saving
        self.parent_dir = ''

        if ck and generator_type=='convKan':
            self.model_name = 'CkkTIM' # Continuous and convolutional KAN Kernel TIM-net

        elif ck and generator_type=='conv':
            self.model_name = 'CkTIM' # Continuous convolutional Kernel TIM-net

        else:
            self.model_name = 'TIM'

    def forward(self, x):
        x1 = self.ctim_net(x)
        out = self.classifier(x1)       
        return out

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