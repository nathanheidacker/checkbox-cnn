# -*- coding: utf-8 -*-
"""
Helper functions for evaluating model performance and visualizing trained
filters. When run as __main__, evaluates the currently saved model on a test
set.
"""

# Standard imports
from __future__ import annotations

# Third Party Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local Imports
from data import CheckboxData
from model import CheckboxCNN

# Typing
from typing import (
    Callable,
    Optional,
)


def evaluate(
    model: nn.Module, data: DataLoader, criterion: Optional[Callable] = None
) -> str:
    model.eval()
    loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss() if criterion is None else criterion

    for features, labels in data:
        preds = model(features)
        loss += criterion(preds, labels)
        preds = preds.argmax(dim=1)
        correct += (preds == labels).sum()

    accuracy = correct / len(data.dataset)
    loss = loss / len(data.dataset)

    return loss.item(), accuracy.item()


def visualize():
    pass


if __name__ == "__main__":
    data = CheckboxData()
    train, test = data.load()
    model = CheckboxCNN("weights.bin")
    loss, accuracy = evaluate(model, test)
    print(loss, accuracy)
