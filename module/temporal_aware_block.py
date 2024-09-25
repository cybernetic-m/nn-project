import torch.nn as nn
import torch
import torch.nn.functional as F
from ckconv import CKConv

class TempAw_Block(nn.Module):

    def __init__(self, dilation_rate, n_filter, kernel_size, cont=False, dropout_rate=0, device='cpu'):

        super(TempAw_Block,self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.cont = cont

        if cont:
            self.conv1 = CKConv(
                input_channels=n_filter,
                output_channels = n_filter,
                device=device
            )

            self.conv2 = CKConv(
                input_channels=n_filter,
                output_channels = n_filter,
                device=device
            )
        
        else:
            self.conv1 = nn.Conv1d(
                in_channels=n_filter,
                out_channels = n_filter,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                device=device
            )

            self.conv2 = nn.Conv1d(
                in_channels=n_filter,
                out_channels = n_filter,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                device=device
            )

        self.batch_norm1 = nn.BatchNorm1d(
            num_features=n_filter, 
            device=device
        )

        self.batch_norm2 = nn.BatchNorm1d(
            num_features=n_filter,
            device=device
        )
        
        self.spatial_drop1 = nn.Dropout1d(
            p=dropout_rate
        )

        self.spatial_drop2 = nn.Dropout1d(
            p=dropout_rate
        )

        self.conv_input = nn.Conv1d(
            in_channels= n_filter,
            out_channels = n_filter,
            kernel_size=1,
            padding='same',
            device=device
        )




    def forward(self, x):
        
        if not self.cont:
            x1 = F.pad(x, ((self.kernel_size-1) * self.dilation_rate, 0)) # Padding Causal 1
        else:
            x1 = x
        #print("Padding Causal1", x2.shape)
        #print("Padding Causal1", x2)

        x2 = self.conv1(x1)
        x2 = self.batch_norm1(x2)
        x2 = F.relu(x2)
        x2 = self.spatial_drop1(x2)
        if not self.cont:
            x2 = F.pad(x2, ((self.kernel_size-1) * self.dilation_rate, 0))  # Padding Causal 2
        x3 = self.conv2(x2)
        x3 = self.batch_norm2(x3)
        x3 = F.relu(x3)
        x3 = self.spatial_drop2(x3)

        x3 = F.sigmoid(x3)

        if x.shape[2] != x3.shape[2]:
            x = self.conv_input(x)
            
        #print("x", x.shape)
        #print("x3", x3.shape)
        y = torch.mul(x,x3)

        return y
    
    
if __name__ == '__main__' :
    nb_filters=64
    kernel_size=2
    dilation_rate=1
    dropout_rate=0.1


    tab = TempAw_Block(
        n_filter=nb_filters, 
        kernel_size=kernel_size, 
        dilation_rate=dilation_rate, 
        dropout_rate=dropout_rate
    )

    t = torch.rand(1,64,10)
    print(t)

    features = tab(t)
