import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Alexish(nn.Module):
    '''Small CNN inspired by AlexNet'''
    def __init__(self,
                 img_channels,
                 num_classes,
                 ):
        super(Alexish, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        self.linear = nn.Sequential(nn.Linear(32 * 14 * 14, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes)
                                    )

    def forward(self, x):

        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.maxpool(self.relu(self.conv4(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

class Alexish5(nn.Module):
    '''Small CNN inspired by AlexNet'''
    def __init__(self,
                 img_channels,
                 num_classes,
                 ):
        super(Alexish5, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(5, 5))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(5, 5))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.tanh = nn.Tanh()

        self.linear = nn.Sequential(nn.Linear(32 * 6 * 6, 512),
                                    nn.Tanh(),
                                    nn.Linear(512, num_classes)
                                    )

    def forward(self, x):

        x = self.maxpool(self.tanh(self.conv1(x)))
        x = self.maxpool(self.tanh(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = self.maxpool(self.tanh(self.conv4(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class PaperCNN(nn.Module):
    '''CNN model used in White Noise paper w/ addition of AvgPool2d'''
    def __init__(self,
                 img_channels,
                 num_classes,
                 ):
        super(PaperCNN, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 20, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.linear = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
