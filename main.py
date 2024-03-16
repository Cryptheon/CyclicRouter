import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import MLP, CMLP, CNN, RouterCNN, RouterMLP
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
from torchviz import make_dot
from cyclic_router_model import load_balancing_loss_func

from utils import save_model

def train(epoch: int, model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, writer: SummaryWriter, args) -> None:
    model.train()
    #global counter 
    #counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        start_time = time.time()

        if args.train_router:
            output, router_logits = model(data)
        else:
            output = model(data)
        
        end_time = time.time()
        if args.train_router:
            loss = criterion(output, target) #+ load_balancing_loss_func(router_logits, 4, 1)
        else:
            loss = criterion(output, target)
        loss.backward()

        #make_dot(output, params=dict(model.named_parameters())).render("torchviz", format="png")

        optimizer.step()
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
        #counter += 1
        if batch_idx % 100 == 0:
            writer.add_scalar('forward_prop_bench/train', np.array(end_time - start_time).mean(), epoch * len(train_loader) + batch_idx)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
    return loss

def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, writer: SummaryWriter, epoch: int, args) -> None:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.train_router:
                output, _ = model(data)
            else:
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

    return accuracy

def main() -> None:

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dataset', type=str, default="mnist", 
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--model_path', type=str, default="./models/cnn_model_trained.pt",
                        help='Model path')
    
    parser.add_argument('--train_router', action='store_true', help='Enable feature')
    parser.add_argument('--save_model', action='store_true', help='Enable feature')

    
    args = parser.parse_args()

    if args.dataset=="mnist":

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

        model = RMLP()

    elif args.dataset=="cifar":

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

        if args.train_router:
            #model = RouterCNN(model_load_path=args.model_path)
            model = RouterMLP(model_load_path=args.model_path)
        else:
            #model = CNN()
            model = MLP()

    # Initialize TensorBoard writer
    writer = SummaryWriter()
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.moeblock.parameters() if args.train_router else model.parameters(), lr=args.lr)

    if args.train_router:
        param_groups = [{'params': model.moeblock.parameters(), 'lr': args.lr},
                        {'params': model.base_model.parameters(), 'lr': 0}]

        optimizer = optim.Adam(param_groups, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(1, args.epochs + 1):
        loss = train(epoch, model, train_loader, criterion, optimizer, writer, args)
        accuracy = test(model, test_loader, criterion, writer, epoch, args)
        
        if args.save_model:
            save_model(model, optimizer, epoch, loss, accuracy, args.model_path)
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
