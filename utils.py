
import torch
import torch.nn as nn

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


def load_trained_model(model, model_path: str) -> nn.Module:
    """
    Load a trained PyTorch model.

    Parameters:
    - model_path (str): Path to the file containing the trained model's state dictionary.

    Returns:
    - model (nn.Module): The loaded model with its trained parameters.
    """

    # Load the state dictionary from the file
    model.load_state_dict(torch.load(model_path)["model_state_dict"])

    return model

def save_model(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Saves the model and optimizer state to a given filepath.

    Parameters:
    - model: The PyTorch model to save.
    - optimizer: The optimizer used during training.
    - epoch: The epoch at which the model is saved.
    - loss: The loss at which the model is saved.
    - filepath: Path to the file where the model and optimizer state are saved.
    """

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, filepath)

    print(f'Model saved to {filepath}')