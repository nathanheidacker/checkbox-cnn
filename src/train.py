# -*- coding: utf-8 -*-
"""
Helper functions for training the model. Trains the model when run as __main__.
"""

# Standard Imports
from __future__ import annotations

# Third Party Imports
import tqdm
import torch
from torch.utils.data import DataLoader

# Local Imports
from .data import CheckboxData
from .model import CheckboxCNN

# Typing:
from typing import Any


def train(
    model: torch.nn.Module,
    data: DataLoader,
    epochs: int = 20,
    criterion=None,
    optimizer=None,
) -> dict[str, torch.Tensor]:
    """
    Trains the passed model for a given number of epochs

    Parameters:
        model:
            The model to be trained

        data:
            The training data to be iterated through in each epoch

        epochs:
            The number of epochs to train for

        criterion:
            The function used to quantify differences between model outputs and
            ground truth labels (the models loss function)

        optimizer:
            The optimization algorithm used to update the model's parameters at
            each training step

    Returns:
        The model's parameters after training is completed
    """
    criterion = torch.nn.CrossEntropyLoss if criterion is None else criterion
    optimizer = torch.optim.Adam(model.parameters) if optimizer is None else optimizer
    for epoch in tqdm(range(epochs)):
        for i, (features, labels) in enumerate(data):
            optimizer.zero_grad()
            predicted = model(features)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


if __name__ == "__main__":
    model = CheckboxCNN()
    data = CheckboxData()
    train, test = data.load()
    weights = train(model, train)
    torch.save(weights, data.path.parent)
