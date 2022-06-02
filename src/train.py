# -*- coding: utf-8 -*-
"""
Helper functions for training the model. Trains the model when run as __main__.
"""

# Standard Imports
from __future__ import annotations
from pathlib import Path
import os
import sys

# Third Party Imports
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Local Imports
from data import CheckboxData
from model import CheckboxCNNv1, CheckboxCNNv2
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
    checkpoint: bool = False,
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

        checkpoint:
            Determines whether the model should save weights every time it
            reaches a new personal best accuracy on the validation set

    Returns:
        The model's parameters after training is completed
    """
    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
    optimizer = torch.optim.Adam(model.parameters()) if optimizer is None else optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

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

    weights_path = Path.joinpath(Path(os.path.abspath(__file__)).parents[1], "weights")
    best_acc = max([int(file[:2])*0.01 for file in os.listdir(weights_path)])
    best_loss = float("inf")
    best_epoch = 0

    # Prints the validation loss and accuracy at every epoch
    if val_data is not None:
        for epoch in range(1, epochs+1):
            with tqdm(
                desc=f"Epoch {epoch} Progress",
                total=len(data.dataset),
                unit="sample",
                ncols=100,
                position=0,
                leave=True,
            ) as pbar:
                for features, labels in data:
                    closure(features, labels)
                    pbar.update(len(labels))
            scheduler.step()

            val_loss, val_acc = evaluate(model, val_data, criterion=criterion)
            print(f"VAL LOSS: {val_loss:.7f} | VAL ACC: {val_acc*100:.2f}%")
            if epoch > 2 and val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch
                if checkpoint:
                    path = Path.joinpath(Path(weights_path), f"{int(val_acc * 100)}.bin")
                    print(f"SAVING MODEL CHECKPOINT TO {path}\n")
                    torch.save(model.state_dict(), path)
                else:
                    print()
            else: print()
            
        print(f"BEST MODEL: {best_acc*100:2f}% ACCURACY @ EPOCH {best_epoch}")
        return model.state_dict()

    # No val data has a single progress bar
    with tqdm(
        desc=f"Training {epochs} epochs",
        total=len(data) * epochs,
        unit="minibatch",
        ncols=100,
        position=0,
        leave=True
    ) as pbar:
        for epoch in range(1, epochs+1):
            for features, labels in data:
                closure(features, labels)
                pbar.update(1)
        scheduler.step()

    return model.state_dict()


if __name__ == "__main__":
    while True:
        model = CheckboxCNNv2()
        data = CheckboxData()
        train_data, test_data = data.load(1000)
        weights = train(model, train_data, test_data, epochs=30, checkpoint=True)
        #weight_path = Path.joinpath(data.path.parent, "weights.bin")
        #torch.save(weights, weight_path)
        #model = CheckboxCNN(weights=weight_path)
