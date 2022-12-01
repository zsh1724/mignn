import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

        self.initialize_weights()

    def initialize_weights(self):
        if self.linear_or_not:
            stdv = 1. / sqrt(self.output_dim)
            self.linear.weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1. / sqrt(self.hidden_dim)
            for layer in range(self.num_layers - 1):
                self.linears[layer].weight.data.uniform_(-stdv, stdv)
            stdv = 1. / sqrt(self.hidden_dim)
            self.linears[-1].weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
            h = self.linears[self.num_layers - 1](h)
            return h