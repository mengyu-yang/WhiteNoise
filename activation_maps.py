import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import os

import utils

# Run trials on noise data samples 
def random_pattern_trials(model, device, total_trials=500000):

    batch_size = 100
    num_trials = int(total_trials/batch_size)
    preds = []
    for _ in range(num_trials):
        noise = torch.rand(batch_size, 1, 28, 28).to(device)
        with torch.no_grad():
            outputs = model(noise)

        pred = torch.argmax(outputs, dim=1)
        preds.append(pred.cpu())

    preds = torch.cat(preds)
    return preds


# Run trials on real data samples
def real_data_trials(model, dataloader, device, total_trials=500000):

    trial_cnt = 0
    preds = []
    while trial_cnt < total_trials:

        for (imgs, label) in dataloader:
            imgs, label = imgs.to(device), label.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            pred = torch.argmax(outputs, dim=1)
            preds.append(pred.cpu())
            trial_cnt += imgs.shape[0]

    preds = torch.cat(preds)
    return preds


def calc_activation_maps(model, hook_conv, device, num_trials, real_data=False, classes=None, dataloader=None):

    activations_list = []

    # Forward hook function that stores the activation maps
    def hook_fcn(self, input, output):
        # Input is stored within a tuple, access by indexing, ie. input = input[0]
        activations_list.append(output.cpu())

    # Register forward hook
    hook_conv.register_forward_hook(hook_fcn)

    # Vector of predicted classes
    if real_data:
        preds = real_data_trials(model, dataloader, device, total_trials=num_trials)
    else:
        preds = random_pattern_trials(model, device, total_trials=num_trials)

    # Matrix of activation maps
    activations = torch.cat(activations_list, dim=0)   # (num_trials, num_filters, feat_map_dim, feat_map_dim)
    activations = activations.mean(1)   # (num_trials, feat_map_dim, feat_map_dim)

    # Average activation maps corresponding to each class
    class_maps = {i: activations[preds == i].mean(dim=0) for i in classes}

    # Plot
    plt.figure(num=None, figsize=(10, 2), dpi=100, facecolor='w', edgecolor='k')
    plt.suptitle(f"Mean Layer Activation Map")
    for i in range(len(classes)):
        plt.subplot(math.ceil(len(classes)/10), len(classes), i + 1)
        plt.axis('off')
        plt.title(f'Class {i}')
        plt.imshow(class_maps[i].numpy())

    plt.show()

    return class_maps


def ActivationMapsMain(ckpt_path, model_name, dataset_name, real_data=False, num_trials=100000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST('./', train=True, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))

    else:
        train_set = torchvision.datasets.FashionMNIST('./', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]))

    dataloader = DataLoader(train_set, batch_size=100, shuffle=True)
    num_classes = len(train_set.targets.unique())
    classes = list(range(num_classes))

    # Load model
    model = utils.load_model(model_name, dataset_name, ckpt_path)
    model = model.to(device)
    model.eval()

    # Get all conv modules from the model
    modules = [module for module in list(model.children()) if isinstance(module, nn.Conv2d)]

    # Average activation maps
    activation_data = "real data" if real_data else "noise data"
    print("*" * 80)
    print(f'Calculating average activation maps for {dataset_name} {model_name} using {activation_data}')
    print("*" * 80)

    act_maps = {}
    for which_conv, hook_conv in zip(["First Conv", "Last Conv"], [modules[0], modules[-1]]):
        print(which_conv)
        act_map = calc_activation_maps(model, hook_conv, device, num_trials=num_trials, real_data=real_data, classes=classes, dataloader=dataloader)
        act_maps[which_conv] = act_map

    return act_maps









