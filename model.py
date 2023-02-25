"""
Define a few models for reinforcement learning.
"""

from torch import nn

class MLP(nn.Module):
    def __init__(self, dims) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i],dims[i+1]) for i in range(len(dims)-1)]
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        return self.layers[-1](x)


    
        