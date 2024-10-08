import torch
from torch import nn
import torch.nn.functional as F  
from torch.nn.utils.parametrizations import weight_norm
from kafnets import KAF

class conv_generator(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_dim, omega_0, dropout_rate, hidden_scale=1, af_type='sin', bias = True, device='cpu'):

        super(conv_generator,self).__init__()

        self.omega_0 = omega_0
        
        self.linear_input = weight_norm(nn.Conv1d(
            input_channels,
            int(hidden_dim*hidden_scale),
            kernel_size=1,
            bias=bias,
            device=device
        ))

        self.linear_hidden = weight_norm(nn.Conv1d(
            int(hidden_dim*hidden_scale),
            int(hidden_dim*hidden_scale),
            kernel_size=1,
            bias=bias,
            device=device
        ))

        self.linear_output = weight_norm(nn.Conv1d(
            int(hidden_dim*hidden_scale),
            output_channels,
            kernel_size=1,
            bias=bias,
            device=device
        ))

        self.batch_norm1 = nn.BatchNorm1d(
            num_features=int(hidden_dim*hidden_scale),
            device=device
        )

        self.batch_norm2 = nn.BatchNorm1d(
            num_features=int(hidden_dim*hidden_scale),
            device=device
        )

        self.dropout = nn.Dropout(
            p=dropout_rate
        )
        self.af_type = af_type
        if af_type == 'KAF':
            self.kaf1 = KAF(int(hidden_dim*hidden_scale), conv=True)
            self.kaf2 = KAF(int(hidden_dim*hidden_scale), conv=True)
        if af_type == 'KAFsin':
            self.kaf1 = KAF(int(hidden_dim*hidden_scale), conv=True, init_fcn=torch.sin)
            self.kaf2 = KAF(int(hidden_dim*hidden_scale), conv=True, init_fcn=torch.sin)

    def forward(self,x):
        #print("Input shape:", x.shape)
        x1 = self.linear_input(x)
        #print("Linear + Weight ->:", x1) 
        #print("Linear + Weight ->:",x1.shape)
        if self.af_type=='sin':
            x1 = self.omega_0 * x1
        #print("Multiply ->:",x1)
        #print("Multiply ->:",x1.shape)
        x1 = self.batch_norm1(x1)
        #print("Norm ->:",x1)
        #print("Norm ->:",x1.shape)
        if self.af_type =='sin':
            x1 = torch.sin(x1)
        if 'KAF' in self.af_type:
            x1 = self.kaf1(x1)
        if self.af_type == 'ReLu':
            x1 = F.relu(x1)
        #print("Activation Function ->:",x1)
        #print("Activation Function ->:",x1.shape)

        x2 = self.linear_hidden(x1)
        #print("Linear + Weight ->:",x2.shape)
        if self.af_type=='sin':
            x2 = self.omega_0 * x2
        #print("Multiply ->:",x2.shape)
        x2 = self.batch_norm2(x2)
        #print("Norm ->:",x2.shape)
        if self.af_type =='sin':
            x2 = torch.sin(x2)
        if 'KAF' in self.af_type:
            x2 = self.kaf1(x2)
        if self.af_type == 'ReLu':
            x2 = F.relu(x2)
        #print("Activation Function ->:",x2.shape)

        x3 = self.linear_output(x2)
        #print("Linear + Weight ->:",x3.shape)
        out = self.dropout(x3)
        #print("Dropout ->:",out)
        #print("Dropout ->:",out.shape)

        return out



