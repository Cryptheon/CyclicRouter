import torch
import torch.nn as nn
from cyclic_router_model import RouterLinear, SparseMoeBlock
from utils import load_trained_model
    
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

class RouterMLP(nn.Module):
    def __init__(self, 
                 hidden_dim: int=512,
                 num_layers: int=4,
                 out_features: int=10, 
                 top_k: int=1, 
                 max_routing: int=2,
                 score_scale_factor: float=1.0,
                 model_load_path: str=None):
        
        super(RouterMLP, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_routing = max_routing
        self.top_k = top_k
        self.num_layers = num_layers

        self.base_model = MLP(hidden_dim=self.hidden_dim, layers=self.num_layers)

        if model_load_path is not None:
            self.base_model = load_trained_model(self.base_model, model_load_path)
            print(f"Model Loaded from {model_load_path}!")

        self.moeblock = SparseMoeBlock(self.hidden_dim,
                                       self.num_layers,
                                       self.top_k,
                                       score_scale_factor)
        
        self.transformator = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        
        x = self.base_model.flatten(x)
        x = self.base_model.input(x)

        router_logits_tuple = []
        for i in range(self.max_routing):
            x, router_logits = self.moeblock(x, self.base_model.layers)
            x = self.transformator(x)
            router_logits_tuple.append(router_logits)

        x = self.base_model.out(x)
        return x, router_logits_tuple

class RouterCNN(nn.Module):
    def __init__(self, 
                 hidden_dim: int=16,
                 num_layers: int=4,
                 out_features: int=10, 
                 top_k: int=1, 
                 max_routing: int=2,
                 score_scale_factor: float=1.0,
                 model_load_path: str=None):
        
        super(RouterCNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_routing = max_routing
        self.top_k = top_k
        self.num_layers = num_layers

        self.base_model = CNN(hidden_dim, num_layers, out_features)

        if model_load_path is not None:
            self.base_model = load_trained_model(self.base_model, model_load_path)
            print(f"Model Loaded from {model_load_path}!")

        self.moeblock = SparseMoeBlock(self.hidden_dim*16*16,
                                       self.num_layers,
                                       self.top_k,
                                       score_scale_factor)
        
        self.transformator = nn.Sequential(nn.Linear(16*16*16, 16*16*16))

    def forward(self, x):
        
        x = self.base_model.first_layer(x)
        B, C, W, H = x.shape
        router_logits_tuple = []
        for i in range(self.max_routing):
            x, router_logits = self.moeblock(x.view(B, -1), self.base_model.cnn_layers)
            x = self.transformator(x)
            router_logits_tuple.append(router_logits)

        x = x.view(B, C, H, W)
        x = self.base_model.pool(self.base_model.last_layer(x))

        x = x.view(-1, (self.hidden_dim//2) * 4 * 4)
        x = self.base_model.relu(self.base_model.fc1(x))
        x = self.base_model.fc2(x)
        return x, router_logits_tuple

class CNN(nn.Module):
    def __init__(self, 
                 hidden_dim: int=16,
                 num_layers: int=4,
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
    def __init__(self, hidden_dim=512, layers=4):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input = nn.Sequential(nn.Linear(32*32*3, hidden_dim), nn.ReLU())

        self.layers = nn.ModuleList(           
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU()) for _ in range(layers)]
        )
        
        self.out = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.input(x)
        for l in self.layers:
            x = l(x)
        x = self.out(x)
        return x