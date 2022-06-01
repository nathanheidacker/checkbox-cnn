# -*- coding: utf-8 -*-
"""
Helper functions for training the model. Trains the model when run as __main__.
"""

# Standard Imports
from __future__ import annotations
from pathlib import Path

# Third Party Imports
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Local Imports
from data import CheckboxData
from model import CheckboxCNN
from evaluate import evaluate

# Typing:
from typing import (
    Optional,
    Callable,
)


def train(
    model: torch.nn.Module,
    data: DataLoader,
    val_data: Optional[DataLoader] = None,
    epochs: int = 1,
    criterion: Optional[Callable] = None,
    optimizer: Optional[Callable] = None,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Trains the passed model for a given number of epochs

    Parameters:
        model:
            The model to be trained

        data:
            The training data to be iterated through in each epoch

        val_data:
            Optional validation data to test validation loss and accuracy at
            the end of every epoch

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
    criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
    optimizer = torch.optim.Adam(model.parameters()) if optimizer is None else optimizer

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        criterion.cuda()

    def closure(features, labels):
        """
        The function responsible for processing the minibatch
        """
        if cuda:
            features, labels = features.cuda(), labels.cuda()
        optimizer.zero_grad()
        predicted = model(features)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()

    # Prints the validation loss and accuracy at every epoch
    if val_data is not None:
        for epoch in range(epochs):
            with tqdm(
                desc="Epoch Progress",
                total=len(data.dataset),
                unit="sample",
                ncols=150,
            ) as pbar:
                for features, labels in data:
                    closure(features, labels)
                    pbar.update(len(labels))

            val_loss, val_acc = evaluate(model, val_data)
            print(f"VAL LOSS: {val_loss:.7f} | VAL ACC: {val_acc*100:.2f}%")

        return model.state_dict()

    # No val data has a single progress bar
    with tqdm(
        desc=f"Training {epochs} epochs",
        total=len(data) * epochs,
        unit="minibatch",
        ncols=150,
    ) as pbar:
        for epoch in range(epochs):
            for features, labels in data:
                closure(features, labels)
                pbar.update(1)

    return model.state_dict()


if __name__ == "__main__":
    model = CheckboxCNN()
    data = CheckboxData()
    train_data, test_data = data.load()
    weights = train(model, train_data, test_data, epochs=5, verbose=True)
    weight_path = Path.joinpath(data.path.parent, "weights.bin")
    torch.save(weights, weight_path)
    model = CheckboxCNN(weights=weight_path)
