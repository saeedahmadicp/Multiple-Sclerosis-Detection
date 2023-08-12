import torch
import torch.nn as nn

__all__ = ['SclerosisClassifier']

class SclerosisClassifier(nn.Module):
    def __init__(self, input_channels, ouptut_units):
        super(SclerosisClassifier, self).__init__()
        
        self.in_channels = input_channels
        self.out_channels = ouptut_units
        
        ## reduce the channels from 512 to 4
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=4, kernel_size=1)
        
        self.ln1 = nn.LayerNorm(normalized_shape=(1051,))
        self.fc1 = nn.Linear(in_features=1051, out_features=512)
        self.ln2 = nn.LayerNorm(normalized_shape=(512,))
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.ln3 = nn.LayerNorm(normalized_shape=(256,))
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.ln4 = nn.LayerNorm(normalized_shape=(128,))
        self.fc4 = nn.Linear(in_features=128, out_features=self.out_channels)
        
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sp_data):
        
        x = self.conv1(x)
        
        ## flatten the features
        x_flatten = x.view(x.size(0), -1)
        
        ### concatenate the flatten features with the supplementory data
        x_new = torch.cat((x_flatten, sp_data), dim=1)
        
        x_new = self.dropout(self.relu(self.fc1(self.ln1(x_new))))
        x_new = self.dropout(self.relu(self.fc2(self.ln2(x_new))))
        x_new = self.dropout(self.relu(self.fc3(self.ln3(x_new))))
        x_new = self.fc4(self.ln4(x_new))
        
        return self.sigmoid(x_new)