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

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(524288, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

        for layer in self.classifier.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(image))


if __name__ == "__main__":
    data = CheckboxData()
    train, test = data.load()
    model = CheckboxCNN()
    criterion = nn.CrossEntropyLoss()
    for X, y in train:
        print(X.shape)
        X = X[:2]
        y = y[:2]
        print(X.shape)

        out = model(X)
        print(out)
        print(criterion(out, y))
        break
