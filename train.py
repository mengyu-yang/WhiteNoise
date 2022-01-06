import torch
import torch.nn as nn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils
import os


def test_eval(testloader, model, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for (imgs, label) in testloader:
            imgs, label = imgs.to(device), label.to(device)
            outputs = model(imgs)

            _, pred = torch.max(outputs, dim=1)

            total += label.shape[0]
            correct += (pred == label).sum().item()

    model.train()
    return 100 * correct / total


def training_loop(model, trainloader, testloader, epochs, device, dataset_name, outdir):
    checkpoint_dir = os.path.join(outdir, 'model_ckpts')
    utils.make_folder(checkpoint_dir)

    optimizer = torch.optim.Adam(model.parameters())
    objective = nn.CrossEntropyLoss()

    model.train()
    best_test_acc = 0

    for epoch in range(epochs):

        loss_accum = 0
        num_batches = 0
        for (imgs, label) in trainloader:
            imgs, label = imgs.to(device), label.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs)

            # Calculate loss, backward pass, optimize
            loss = objective(outputs, label)
            loss.backward()
            optimizer.step()

            loss_accum += loss.data
            num_batches += 1

        # Print info after every epoch
        print(f'Epoch {epoch + 1}/{epochs} | Loss: {loss_accum / num_batches}')

        # Evaluate
        test_acc = test_eval(testloader, model, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            fname = f'{model.__class__.__name__}_{dataset_name}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_test_acc
            }, os.path.join(checkpoint_dir, fname))
            print(f'Saved best model, current test acc is {best_test_acc}%')


def train_main(outdir, dataset_name, model_name, batch_size, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(outdir, train=True, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))

        test_set = torchvision.datasets.MNIST(outdir, train=False, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
    else:
        train_set = torchvision.datasets.FashionMNIST(outdir, train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]))

        test_set = torchvision.datasets.FashionMNIST(outdir, train=False, download=True,
                                                     transform=transforms.Compose([transforms.ToTensor()]))

    loader_kwargs = dict(batch_size=batch_size, shuffle=True)
    trainloader = DataLoader(train_set, **loader_kwargs)
    testloader = DataLoader(test_set, **loader_kwargs)

    # Load model
    common_kwargs = dict(class_name=f'models.{model_name}', img_channels=1, num_classes=len(train_set.targets.unique()))
    model = utils.construct_class_by_name(**common_kwargs)
    model = model.to(device)

    # Train
    print("*" * 50)
    print(f'Training {model_name} on {dataset_name}')
    print("*" * 50)
    training_loop(model, trainloader, testloader, epochs, device, dataset_name, outdir)

