# -*- coding: utf-8 -*-
"""
The model for classifying checked boxes
"""

# Standard Imports
from __future__ import annotations

# Third Party Imports
import torch
import torch.nn as nn

# Local Imports
from data import CheckboxData, PathLike

# Typing
from typing import Optional

class CheckboxCNN(torch.nn.Module):
    def _init_weights(self, weights: Optional[PathLike] = None) -> None:
        if weights:
            loaded = None
            if torch.cuda.is_available():
                loaded = torch.load(weights, map_location=torch.device('cuda'))
            else:
                loaded = torch.load(weights, map_location=torch.device('cpu'))
            self.load_state_dict(loaded)
        else:
            # Conv2D layers use xavier normal initialization
            for layer in self.conv.children():
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()

            # Linear layers use xavier uniform
            for layer in self.classifier.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                elif isinstance(layer, nn.BatchNorm1d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()

    def visualize(self, layer):
        return


class CheckboxCNNv1(CheckboxCNN):
    """
    A Convolutional Neural network designed to detect and classify the state of
    HTML checkboxes

    Parameters:
        weights:
            An optional parameter specifying a path to trained weights for this
            model
    """

    def __init__(self, weights: Optional[PathLike] = None) -> None:
        super().__init__()

        # The convolutional forward pass
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification of convolved features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(131072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 3),
        )

        # Loading/initializing weights
        self._init_weights(weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(image))


class CheckboxCNNv2(CheckboxCNN):
    """
    A Convolutional Neural network designed to detect and classify the state of
    HTML checkboxes

    Parameters:
        weights:
            An optional parameter specifying a path to trained weights for this
            model
    """

    def __init__(self, weights: Optional[PathLike] = None) -> None:
        super().__init__()

        # The convolutional forward pass
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification of convolved features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 3),
        )

        # Loading/initializing weights
        self._init_weights(weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(image))
