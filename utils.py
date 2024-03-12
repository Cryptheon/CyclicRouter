
import torch

def load_pretrained_model():
    pass

def write_grad_flow(named_parameters, writer):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #print(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
            writer.add_scalar(f'grad_average/{n}', p.grad.abs().mean().item(), counter)
            writer.add_scalar(f'grad_norm/{n}', p.grad.norm().item(), counter)
            writer.add_scalar(f'grad_max/{n}', p.grad.max().item(), counter)