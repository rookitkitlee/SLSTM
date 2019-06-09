import torch
import torch.nn as nn

class classifer(nn.Module):

    def __init__(self, base_size):
        super(classifer,self).__init__()
        self.cc = nn.Sequential(
            nn.Linear(base_size,base_size*2),
            nn.ReLU(),
            nn.Linear(base_size*2,base_size),
            nn.ReLU(),
            nn.Linear(base_size,2)
        )

    def forward(self, x):
        out = self.cc(x)
        return torch.sigmoid(out)