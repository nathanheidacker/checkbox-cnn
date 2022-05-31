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
    criterion = torch.nn.CrossEntropyLoss if criterion is None else criterion
    optimizer = torch.optim.Adam(model.parameters)
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
