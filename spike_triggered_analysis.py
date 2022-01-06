import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pytorch_receptive_field.torch_receptive_field import receptive_field, receptive_field_for_unit

import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

# Run trials on noise data samples, storing each sample cropped to the receptive field
def random_pattern_trials(model, coords, device, total_trials):

    rf_h1, rf_h2, rf_w1, rf_w2 = coords
    batch_size = 100
    num_trials = int(total_trials/batch_size)
    noises = []
    for _ in range(num_trials):
        noise = torch.rand(batch_size, 1, 28, 28).to(device)
        noises.append(noise[..., rf_h1:rf_h2, rf_w1:rf_w2].cpu())
        with torch.no_grad():
            _ = model(noise)

    return noises

# Run trials on real data samples, storing each sample cropped to the receptive field
def real_data_trials(model, coords, dataloader, device, total_trials):

    rf_h1, rf_h2, rf_w1, rf_w2 = coords
    trial_cnt = 0
    img_rfs = []
    while trial_cnt < total_trials:

        for (imgs, label) in dataloader:
            imgs, label = imgs.to(device), label.to(device)
            img_rfs.append(imgs[..., rf_h1:rf_h2, rf_w1:rf_w2].cpu())
            with torch.no_grad():
                _ = model(imgs)

            trial_cnt += imgs.shape[0]

    return img_rfs

def STA(model, hook_conv, coords, center_coord, device, num_trials, real_data=False, dataloader=None):

    activations_list = []

    rf_h1, rf_h2, rf_w1, rf_w2 = coords

    # The forward hook function stores the activation value corresponding to center_coord
    def hook_fcn(self, input, output):
        activations_list.append(output[..., center_coord, center_coord].cpu())

    # Forward hook
    hook_conv.register_forward_hook(hook_fcn)

    # Run trials
    if real_data:
        receptive_fields = real_data_trials(model, coords, dataloader, device, total_trials=num_trials)
    else:
        receptive_fields = random_pattern_trials(model, coords, device, total_trials=num_trials)

    # Vector of activations
    activations = torch.cat(activations_list, dim=0).unsqueeze(-1)

    # Vector of (flattened) receptive fields
    receptive_fields = torch.cat(receptive_fields, dim=0)
    receptive_fields = receptive_fields.view(-1, 1, (rf_h2 - rf_h1) * (rf_w2 - rf_w1))

    # Receptive fields average weighted by the corresponding activation values
    noise_maps = (activations * receptive_fields).mean(dim=0).view(-1, (rf_h2 - rf_h1), (rf_w2 - rf_w1))

    # Plot spike triggered averaging filters
    plt.figure(num=None, figsize=(10, 4.5), dpi=100, facecolor='w', edgecolor='k')
    plt.suptitle("Spike Triggered Averaging Filters")
    for i in range(noise_maps.shape[0]):
        plt.subplot(7, 10, i + 1)
        plt.axis('off')
        plt.imshow(noise_maps[i].cpu().numpy())

    plt.show()


def STA_main(ckpt_path, model_name, dataset_name, num_trials=100000, real_data=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST('./', train=True, download=True,
                                               transform=transforms.Compose([transforms.ToTensor()]))

    else:
        train_set = torchvision.datasets.FashionMNIST('./', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor()]))

    dataloader = DataLoader(train_set, batch_size=100, shuffle=True)

    # Load model
    model = utils.load_model(model_name, dataset_name, ckpt_path)
    model = model.to(device)
    model.eval()

    # Initialize function for calculating receptive field
    receptive_field_dict = receptive_field(model, (1, 28, 28))

    # Get layer index of the conv layers
    conv_layers = []
    for layer in receptive_field_dict:
        if len(layer) == 1:
            if receptive_field_dict[layer]['name'] == 'Conv2d':
                conv_layers.append(layer)

    # Get all conv modules from the model
    modules = [module for module in list(model.children()) if isinstance(module, nn.Conv2d)]

    # Average activation maps
    activation_data = "real data" if real_data else "noise data"
    print("*" * 80)
    print(f'Calculating spike triggered averaging for {dataset_name} {model_name} using {activation_data}')
    print("*" * 80)

    for which_conv, hook_conv, conv_idx in zip(["First Conv", "Second Conv"], [modules[0], modules[1]], conv_layers):
        print(which_conv)

        '''
        Instead of averaging across all windows seen by the conv filter, we will only extract one window for each 
        sample and average across the samples. We will take approximately the center window, and center_coord is 
        the corresponding coordinate on the feature map. 
        '''
        center_coord = receptive_field_dict[conv_idx]['output_shape'][-1] // 2

        '''
        rf_h1: bottom coordinate of receptive field 
        rf_h2: top coordinate of receptive field 
        rf_w1: left coordinate of receptive field
        rf_w2: right coordinate of receptive field
        '''
        rf = receptive_field_for_unit(receptive_field_dict, conv_idx, (center_coord, center_coord))
        rf_h1 = int(rf[0][0])
        rf_h2 = int(rf[0][1])
        rf_w1 = int(rf[1][0])
        rf_w2 = int(rf[1][1])
        coords = (rf_h1, rf_h2, rf_w1, rf_w2)

        STA(model, hook_conv, coords, center_coord, device, num_trials=num_trials, real_data=real_data, dataloader=dataloader)





















