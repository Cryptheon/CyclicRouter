import torch
from torch import Tensor
import math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import time

class MixtralSparseMoeBlock(nn.Module):
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

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.score_scale_factor = config.score_scale_factor

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
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
            expert_layer = self.experts[expert_idx]
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
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
                * self.score_scale_factor
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

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
                 bias: bool = True,
                 top_k: int = 64,
                 device=None, 
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.top_k = top_k
        self.select = top_k > 0
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, 
                input: Tensor, 
                previously_selected_nodes: Tensor=None) -> Tensor:

        if previously_selected_nodes is not None:
             # of shape (Batch, Input_dimension, 1)
            input = input.unsqueeze(-1)
            B, in_features = previously_selected_nodes.shape
            #weight = self.weight[:, previously_selected_nodes].permute(1,0,2)

            weight = torch.index_select(self.weight, 1, previously_selected_nodes.flatten())
            weight = weight.view(self.out_features, B, in_features).permute(1,0,2)

            result = torch.bmm(weight, input).squeeze(-1) + self.bias
        
        else:
            result = F.linear(input, self.weight, self.bias)

        #routing_weights = F.softmax(result, dim=-1)

        if self.select:
            
            _, selected_nodes = torch.topk(result, self.top_k, dim=-1)

            result = torch.gather(result, 1, selected_nodes)
            
            return result, selected_nodes
        
        return result, None
    

