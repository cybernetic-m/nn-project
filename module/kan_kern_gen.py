#just a test

import torch
from torch import nn
import torch.nn.functional as F  
from torch.nn.utils import weight_norm  
from kan import multKAN as KAN

class kan_generator(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_dim, hidden_dim, omega_0 , dropout_rate, bias = True):

        super(kan_generator, self).__init__()

        self.kernel_dim = kernel_dim
        self.output_channels = output_channels
        self.omega_0 = omega_0
        
        self.linear_input = weight_norm(KAN(
            [input_channels,
            hidden_dim],
            bias=bias
        ))

        self.linear_hidden = weight_norm(KAN(
            [hidden_dim,
            hidden_dim],
            bias=bias
        ))

        self.linear_output = weight_norm(KAN(
            [hidden_dim,
            output_channels*kernel_dim],
            bias=bias
        ))

        self.instance_norm1 = nn.InstanceNorm1d(
            num_features=hidden_dim
        )

        self.instance_norm2 = nn.InstanceNorm1d(
            num_features=hidden_dim
        )

        self.dropout = nn.Dropout(
            p=dropout_rate
        )


    def forward(self,x):
        x1_flatten = x.view(x.shape[0], x.shape[1]*x.shape[2])   # Flattening Ex. [10, 2, 100] -> [10, 200] Mantain Batch_Size
        print("Flattening ->:",x1_flatten.shape)
        x1 = self.linear_input(x1_flatten) 
        print("Linear + Weight ->:",x1.shape)
        x1 = self.omega_0 * x1
        print("Multiply ->:",x1.shape)
        #x1 = self.instance_norm1(x1)
        #print("Norm ->:",x1.shape)
        x1 = torch.sin(x1)
        print("Activation Function ->:",x1.shape)

        x2 = self.linear_hidden(x1)
        print("Linear + Weight ->:",x2.shape)
        x2 = self.omega_0 * x2
        print("Multiply ->:",x2.shape)
        #x2 = self.instance_norm2(x2)
        #print("Norm ->:",x2.shape)
        x2 = torch.sin(x2)
        print("Activation Function ->:",x2.shape)

        x3 = self.linear_output(x2)
        print("Linear + Weight ->:",x3.shape)
        out_flatten = self.dropout(x3)
        out = out_flatten.view(out_flatten.shape[0], self.output_channels, self.kernel_dim)
        print("Dropout ->:",out.shape)

        return out
    

kernel_dim = 10

tensor = torch.randn(1,2,100)

rel_positions = (torch.linspace(-1.0, 11, kernel_dim).unsqueeze(0).unsqueeze(0))

print(rel_positions)

kernel_gen = mlp_generator(
        input_channels = kernel_dim,
        output_channels = 2,
        hidden_dim = 32,
        kernel_dim=kernel_dim,
        omega_0 = 1,
        dropout_rate = 0.5,
        bias= False,
    )

out = kernel_gen(rel_positions)

print(out)



