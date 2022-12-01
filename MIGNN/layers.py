import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from mlp import MLP

class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, radius):
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius
        self.fc1 = nn.ModuleList([MLP(2, input_dim, output_dim, output_dim, dropout) for i in range(radius+1)])
        self.fc2 = nn.ModuleList([MLP(2, output_dim, output_dim, output_dim, dropout) for i in range(radius)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        l = list()
        for i in range(self.radius+1):
            l.append(features[i])
        for i in range(self.radius-1, -1, -1):
            if i==self.radius-1:
                x = self.fc1[i+1](l[i+1])
                x = self.fc1[i](l[i]) + torch.spmm(adj[i], self.fc2[i](x))
            else:
                x = self.fc1[i](l[i]) + torch.spmm(adj[i], self.fc2[i](x))
            x = self.dropout(x)
            
        return x
