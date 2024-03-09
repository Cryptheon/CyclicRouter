import torch
import torch.nn as nn
from cyclic_router_model import RouterLinear

class RMLP(nn.Module):
    def __init__(self, 
                 in_features: int=784, 
                 out_features: int=10, 
                 hidden_dim: int=32, 
                 num_layers: int=4, 
                 top_k: int=1, 
                 max_routing: int=4,
                 score_scale_factor: float=0.95):
        
        super(RMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.router_network = RouterLinear(in_features,
                     out_features,
                     hidden_dim,
                     num_experts=num_layers,
                     top_k=top_k,
                     max_routing=max_routing,
                     score_scale_factor=score_scale_factor)
                     

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.router_network(x)
        return x
    
class CMLP(nn.Module):
    def __init__(self, 
                 in_features: int=784, 
                 out_features: int=10, 
                 hidden_dim: int=32):
        
        super(CMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(in_features, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l2(x)
        x = self.l2(x)
        x = self.output(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_dim=32, layers=4):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input= nn.Linear(28*28, hidden_dim)
        self.layers = nn.ModuleList(           
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU()) for _ in range(layers)]
        )
        
        self.out = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.relu(self.input(x))
        for l in self.layers:
            x = l(x)
        x = self.out(x)
        return x