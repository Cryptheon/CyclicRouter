import torch
import torch.nn as nn
from cyclic_router_model import RouterLinear

class CRMLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 1024)
        self.layer2 = nn.Linear(1024, 32)
        self.layer3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x
