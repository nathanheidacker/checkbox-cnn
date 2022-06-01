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
) -> tuple(float, float):
    """
    Evaluates the model's performance by calculating loss and accuracy on a set
    of test data

    Parameters:
        model:
            The trained model to evaluate

        data:
            The data to test accuracy/loss of

        criterion:
            The loss function to utilize

    Returns:
        A tuple of floats, with the first index corresponding to loss and the
        second to accuracy
    """
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss() if criterion is None else criterion

        loss = 0
        correct = 0
        cuda = torch.cuda.is_available()

        if cuda:
            model.cuda()
            criterion.cuda()

        for features, labels in data:
            if cuda:
                features, labels = features.cuda(), labels.cuda()
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
