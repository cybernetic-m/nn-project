import torch.nn as nn
import torch
import torch.nn.functional as F
from ckconv import CKConv

class TempAw_Block(nn.Module):

    def __init__(self, dilation_rate, n_filter, kernel_size, is_siren, omega_0=1, ck=False, dropout_rate=0.1, device='cpu'):

        super(TempAw_Block,self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.ck = ck

        if self.ck:
            self.conv1 = CKConv(
                input_channels=n_filter,
                output_channels = n_filter,
                omega_0=omega_0,
                is_siren=is_siren,
                dropout_rate=dropout_rate,
                device=device
            )

            self.conv2 = CKConv(
                input_channels=n_filter,
                output_channels = n_filter,
                omega_0=omega_0,
                is_siren=is_siren,
                dropout_rate=dropout_rate,
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
        '''
        self.conv_input = nn.Conv1d(
            in_channels= n_filter,
            out_channels = n_filter,
            kernel_size=1,
            padding='same',
            device=device
        )
        '''

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        if not self.ck:
            x1 = F.pad(x, ((self.kernel_size-1) * self.dilation_rate, 0)) # Padding Causal 1
        else:
            x1 = x
        #print("Padding Causal1", x2.shape)
        #print("Padding Causal1", x2)
        #print("x1", x1.shape)
        x2 = self.conv1(x1)
        #print("x2 conv",x2.shape)
        x2 = self.batch_norm1(x2)
        #print("x2 btach norm",x2.shape)
        x2 = self.relu1(x2)
        #print("x2 relu",x2.shape)
        x2 = self.spatial_drop1(x2)
        #print("x2 spatial drop",x2.shape)
        
        if not self.ck:
            x2 = F.pad(x2, ((self.kernel_size-1) * self.dilation_rate, 0))  # Padding Causal 2
        
        #print("x2", x2.shape)
        x3 = self.conv2(x2)
        #print("x3 conv",x3.shape)
        x3 = self.batch_norm2(x3)
        #print("x3 batch norm",x3.shape)
        x3 = self.relu2(x3)
        #print("x3 relu",x3.shape)
        x3 = self.spatial_drop2(x3)
        #print("x3 spatial drop",x3.shape)

        x3 = self.sigmoid(x3)
        #print("x3 sigmoid",x3.shape)
        '''
        if x.shape[1] != x3.shape[1]:
            print("check check prova 1 2 3")
            x = self.conv_input(x)
        '''
            
        #print("x", x.shape)
        y = torch.mul(x,x3)
        #print("x3 final", x3.shape)

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
