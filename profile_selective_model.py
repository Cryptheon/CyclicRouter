import torch
from torch import Tensor
import math
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

class SelectiveLinear(nn.Module):
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
                 top_k: int = 2,
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
            with record_function("unsqueeze"):
                # of shape (Batch, Input_dimension, 1)
                input = input.unsqueeze(-1)

            with record_function("index_select"):
                #weight = self.weight[:, previously_selected_nodes.tolist()]
                B, in_features = previously_selected_nodes.shape
                #print("B", B, "in", in_features)
                weight = torch.index_select(self.weight, 1, previously_selected_nodes.flatten())
                #print("weight shape", weight.shape)
            
            with record_function("permute"):
                weight = weight.view(B, self.out_features, in_features)
                #weight = weight.permute(1,0,2)
                #print(weight.shape)

            with record_function("torch_bmm"):
                result = torch.bmm(weight, input).squeeze(-1) + self.bias
        
        else:
            with record_function("F_linear"):
                result = F.linear(input, self.weight, self.bias)

        #routing_weights = F.softmax(result, dim=-1)

        if self.select:
            
            with record_function("topk"):
                _, selected_nodes = torch.topk(result, self.top_k, dim=-1)

            with record_function("gather"):
                result = torch.gather(result, 1, selected_nodes)
            
            return result, selected_nodes
        
        return result, None
    
writer = SummaryWriter("runs/profiling_SL")
model = SelectiveLinear(2048,2048, top_k=64)
inp = torch.randn(64,2048)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True) as prof:
   with record_function("model_forward"):
    output, select = model(inp)

profiling_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

writer.add_text("Profiling Results forward 1", profiling_results)

prof.export_chrome_trace("trace.json")  # Export the trace to a file
writer.close()

writer = SummaryWriter("runs/profiling_SL")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True) as prof:
   with record_function("model_forward_2"):
    output, _ = model(output, select)

profiling_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)

writer.add_text("Profiling Results forward 2", profiling_results)

prof.export_chrome_trace("trace_2.json")  # Export the trace to a file
writer.close()


