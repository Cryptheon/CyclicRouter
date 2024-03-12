import torch
import torch.nn as nn
from cyclic_router_model import RouterLinear, SparseMoeBlock

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


class RouterCNN(nn.Module):
    def __init__(self, 
                 hidden_dim: int=16,
                 num_layers: int=6,
                 out_features: int=10, 
                 top_k: int=1, 
                 max_routing: int=4,
                 score_scale_factor: float=0.95):
        
        super(RouterCNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_routing = max_routing
        self.top_k = top_k
        self.num_layers = num_layers

        self.first_layer = nn.Sequential(nn.Conv2d(3, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU())

        self.cnn_layers = nn.ModuleList(           
            [nn.Sequential(nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1), nn.ReLU()) for _ in range(self.num_layers)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear((self.hidden_dim//2) * 4 * 4, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, out_features)
        self.relu = nn.ReLU()

        self.moeblock = SparseMoeBlock(self.hidden_dim*16*16,
                                       self.num_layers,
                                       self.top_k,
                                       score_scale_factor)

        self.last_layer = nn.Sequential(nn.Conv2d(self.hidden_dim, self.hidden_dim//2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU())

    def forward(self, x):
        
        x = self.first_layer(x)
        B, C, W, H = x.shape
        for i in range(self.max_routing):
            x, router_logits = self.moeblock(x.view(B, -1), self.cnn_layers)

        x = x.view(B, C, H, W)
        x = self.pool(self.last_layer(x))

        x = x.view(-1, (self.hidden_dim//2) * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, 
                 hidden_dim: int=32,
                 num_layers: int=6,
                 out_features: int=10):
        
        super(CNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.first_layer = nn.Sequential(nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU())

        self.cnn_layers = nn.ModuleList(           
            [nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), nn.ReLU()) for _ in range(num_layers)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear((hidden_dim//2) * 4 * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.relu = nn.ReLU()

        self.last_layer = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU())

    def forward(self, x):
        
        x = self.first_layer(x)
        for layer in self.cnn_layers:
            x = layer(x)

        x = self.pool(self.last_layer(x))

        x = x.view(-1, (self.hidden_dim//2) * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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