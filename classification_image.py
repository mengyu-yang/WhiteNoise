import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from models import Alexish5, PaperCNN
import inspect
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import os

import utils


def CalcAverageNoiseMap(model, dataset_name, classes, num_trials, device):
    model.eval()

    batch_size = 100
    num_trials = int(num_trials / batch_size)
    preds = []
    noises = []
    for _ in range(num_trials):
        noise = torch.rand(batch_size, 1, 28, 28).to(device)
        noises.append(noise.cpu())
        with torch.no_grad():
            outputs = model(noise)

        pred = torch.argmax(outputs, dim=1)
        preds.append(pred.cpu())

    preds = torch.cat(preds)
    noises = torch.cat(noises, dim=0)

    noise_maps = {i: noises[preds == i].mean(dim=0).squeeze() if noises[preds == i].shape[0] > 0 else torch.zeros(28, 28).float() for i in classes}


    plt.figure(num=None, figsize=(10, 2), dpi=100, facecolor='w', edgecolor='k')
    plt.suptitle(f'Classification Images for {model.__class__.__name__} on {dataset_name}')
    for i in range(len(classes)):
        plt.subplot(math.ceil(len(classes)/10), len(classes), i + 1)
        plt.axis('off')
        plt.title(f'Class {i}')
        plt.imshow(noise_maps[i].numpy())
    plt.show()

    return noise_maps


def NoiseMapClassification(noise_maps, dataloader):
    labels = []
    preds = []
    for (img, label) in dataloader:

        # Repeat the image to same shape as noise_maps (n_classes, 28, 28)
        img = img.squeeze().repeat(noise_maps.shape[0], 1, 1)

        # Dot product along dims=1,2
        dot_prods = torch.sum(img * noise_maps, dim=[1, 2])

        # Prediction corresponds to largest dot product
        pred = torch.argmax(dot_prods)

        labels.append(label.cpu().item())
        preds.append(pred.cpu().item())


    confuse = (confusion_matrix(labels, preds, normalize='true'))
    disp = ConfusionMatrixDisplay(confuse)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.title("Classification using Noise Maps")
    plt.show()


def ClassificationImgMain(ckpt_dir, model_name, dataset_name, num_trials=1000000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST('./', train=True, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))

    else:
        train_set = torchvision.datasets.FashionMNIST('./', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]))

    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    num_classes = len(train_set.targets.unique())
    classes = list(range(num_classes))

    # Load model
    model = utils.load_model(model_name, dataset_name, ckpt_dir)
    model = model.to(device)
    model.eval()

    # Calculate classification images for each class
    print("*" * 80)
    print(f'Calculating classification images for {model_name} trained on {dataset_name}')
    print("*" * 80)
    noise_maps = CalcAverageNoiseMap(model, dataset_name, classes, num_trials, device)

    # Classify the noise maps
    preds = []
    for i in classes:
        noise_map = noise_maps[i].unsqueeze(0).unsqueeze(0).to(device)
        output = model(noise_map)
        pred = torch.argmax(output)
        preds.append(pred.cpu().data)

    confuse = (confusion_matrix(classes, preds, normalize='true'))
    disp = ConfusionMatrixDisplay(confuse)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.title("Classification of Noise Maps")
    plt.show()

    # Use noise maps as classifier

    # Concat all noise maps into a tensor of size (n_noise_maps, noise_map_dim, noise_map_dim)
    noise_maps = list(noise_maps.values())
    noise_maps = torch.stack(noise_maps, dim=0)

    print('\n')
    print(f"Using average noise maps as classifiers")
    NoiseMapClassification(noise_maps, dataloader)

