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
import cv2

import utils


def predict(model, img):
    output = model(img)
    pred = torch.argmax(output)
    return pred

def normalize(data):
    data -= data.min().item()
    data /= data.max().item()
    return data


def add_bias_map(data, biasmap):
    # Normalize
    data = normalize(data)
    biasmap = normalize(biasmap)

    # Add and normalize result
    bias_up = cv2.resize(biasmap.cpu().numpy(), (28, 28))
    data += torch.from_numpy(bias_up).to('cuda')
    data = normalize(data)
    return data

def AdversarialMain(ckpt_path, model_name, dataset_name, ref_class, bias_class, act_map, which_layer):

    print("*" * 80)
    print(f"Adversarial experiment using {model_name} on {dataset_name}...")
    print("*" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST('./', train=True, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))

    else:
        train_set = torchvision.datasets.FashionMNIST('./', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]))

    # Load model
    model = utils.load_model(model_name, dataset_name, ckpt_path)
    model = model.to(device)
    model.eval()

    # Get all images in dataset belonging to the reference class, then select one (let's take the fist one)
    ref_class_imgs = train_set.data[train_set.targets == ref_class].unsqueeze(1).to(device).float()
    ref_img = ref_class_imgs[0]

    # Visualize the reference image and report its predicted class
    pred = predict(model, ref_img.unsqueeze(0))
    print(f"Original image for reference class {ref_class}, with predicted class {pred}: ")
    plt.imshow(ref_img.squeeze().cpu().numpy())
    plt.show()

    # Get the activation map for the adversarial class
    adversarial_map = act_map[model_name + dataset_name][which_layer][bias_class].to(device)
    adversarial_map = normalize(adversarial_map)
    # act_map0 = 1 - act_map0

    # Visualize the adversarial activation map
    print(f"Mean layer activation map of class {bias_class}: ")
    plt.imshow(adversarial_map.squeeze().cpu().numpy())
    plt.show()

    # Add the adversarial map to the original image. Visualize and calculate predicted class
    combined = add_bias_map(ref_img, adversarial_map)

    pred = predict(model, combined.unsqueeze(0))
    print(f"Combined image, with predicted class {pred}: ")
    plt.imshow(combined.squeeze().cpu().numpy())
    plt.show()








