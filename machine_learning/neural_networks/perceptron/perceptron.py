import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_size: int):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

