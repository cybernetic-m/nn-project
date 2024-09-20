import torch
from torch import nn
import torch.nn.functional as F  
from torch.nn.utils import weight_norm  

class conv_generator(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_dim, omega_0 , dropout_rate, bias = True):

        super(conv_generator,self).__init__()

        self.omega_0 = omega_0
        
        self.linear_input = weight_norm(nn.Conv1d(
            input_channels,
            hidden_dim,
            kernel_size=1,
            bias=bias
        ))

        self.linear_hidden = weight_norm(nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            bias=bias
        ))

        self.linear_output = weight_norm(nn.Conv1d(
            hidden_dim,
            output_channels,
            kernel_size=1,
            bias=bias
        ))

        self.batch_norm1 = nn.BatchNorm1d(
            num_features=hidden_dim
        )

        self.batch_norm2 = nn.BatchNorm1d(
            num_features=hidden_dim
        )

        self.dropout = nn.Dropout(
            p=dropout_rate
        )


    def forward(self,x):
        #print("Input shape:", x.shape)
        x1 = self.linear_input(x)
        #print("Linear + Weight ->:", x1) 
        #print("Linear + Weight ->:",x1.shape)
        x1 = self.omega_0 * x1
        #print("Multiply ->:",x1)
        #print("Multiply ->:",x1.shape)
        x1 = self.batch_norm1(x1)
        #print("Norm ->:",x1)
        #print("Norm ->:",x1.shape)
        x1 = torch.sin(x1)
        #print("Activation Function ->:",x1)
        #print("Activation Function ->:",x1.shape)

        x2 = self.linear_hidden(x1)
        #print("Linear + Weight ->:",x2.shape)
        x2 = self.omega_0 * x2
        #print("Multiply ->:",x2.shape)
        x2 = self.batch_norm2(x2)
        #print("Norm ->:",x2.shape)
        x2 = torch.sin(x2)
        #print("Activation Function ->:",x2.shape)

        x3 = self.linear_output(x2)
        #print("Linear + Weight ->:",x3.shape)
        out = self.dropout(x3)
        #print("Dropout ->:",out)
        #print("Dropout ->:",out.shape)

        return out



