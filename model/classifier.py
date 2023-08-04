import torch
import torch.nn as nn

__all__ = ['SclerosisClassifier']

class SclerosisClassifier(nn.Module):
    def __init__(self, input_units, ouptut_units):
        super(SclerosisClassifier, self).__init__()
        
        self.in_channels = input_units
        self.out_channels = ouptut_units
        
        self.fc1 = nn.Linear(in_features=self.filters[-1], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=self.out_channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc3(x)
        
        return x