import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import MLP, RMLP, CMLP
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from torchviz import make_dot

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

def train(epoch: int, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, writer: SummaryWriter) -> None:
    model.train()
    global counter 
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        start_time = time.time()

        output = model(data)
        
        end_time = time.time()
        loss = criterion(output, target)
        loss.backward()

        #make_dot(output, params=dict(model.named_parameters())).render("torchviz", format="png")
        #if batch_idx==50:
        #    raise ValueError()
        #raise ValueError()
        #write_grad_flow(model.named_parameters(), writer)
        optimizer.step()
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        counter += 1
        if batch_idx % 100 == 0:
            writer.add_scalar('forward_prop_bench/train', np.array(end_time - start_time).mean(), epoch * len(train_loader) + batch_idx)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, writer: SummaryWriter, epoch: int) -> None:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # Log test loss and accuracy to TensorBoard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

def main() -> None:

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    model = RMLP()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, writer)
        test(model, test_loader, criterion, writer, epoch)
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
