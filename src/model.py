import torch
import torch.nn as nn


class SimpleMLP(nn.Module):  #定义一个SimpleMLP类，继承自nn.Module
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)

        return x