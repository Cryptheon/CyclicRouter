import torch
from torch import Tensor
import math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import time

# UNUSED
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the non-differentiable operation in the forward pass
        # input here is expected to be the selected_nodes tensor
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, we simply pass the gradients through unchanged
        return grad_output.clone()

def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: int, top_k: int=1) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
            
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """

    
    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class SparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, 
                 hidden_dim: int,
                 num_experts: int,
                 top_k: int,
                 score_scale_factor: int
                 ):
        super().__init__()
        self.top_k = top_k
        self.score_scale_factor = score_scale_factor
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # gating
        self.gate = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim//4), nn.ReLU(), 
                                  nn.Linear(self.hidden_dim//4, self.hidden_dim//16), nn.ReLU(),
                                  nn.Linear(self.hidden_dim//16, self.num_experts, bias=False))

    def forward(self, 
                hidden_states: torch.Tensor,
                experts: nn.ModuleList) -> torch.Tensor:
        """ """
        batch_size, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # print("selected experts: ", selected_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # Necessary for CNNs
            B, _ = current_state.shape
            #current_state = current_state.view(B, 16, 16, 16)
            current_hidden_states = (
                expert_layer(current_state)#.view(B, -1)
                * routing_weights[top_x_list, idx_list, None]
                * self.score_scale_factor
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, hidden_dim
        )
        return final_hidden_states, router_logits

class RouterLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 hidden_dim: int, 
                 top_k: int = 1,
                 max_routing: int = 1,
                 num_experts: int = 1,
                 score_scale_factor: float = 0.75,
                 device=None, 
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.max_routing = max_routing
        self.num_experts = num_experts

        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_features)

        self.experts = nn.ModuleList(           
            [nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.ReLU()) for _ in range(self.num_experts)]
        )

        self.moeblock = SparseMoeBlock(self.hidden_dim,
                                       self.num_experts,
                                       self.top_k,
                                       score_scale_factor)

    def forward(self, input: Tensor) -> Tensor:
        
        x = torch.relu(self.input_layer(input))
        for i in range(self.max_routing):
            x, router_logits = self.moeblock(x, self.experts)
        
        result = self.output_layer(x)
        
        return result
    

