# -*- coding: utf-8 -*-
"""
Helper functions for evaluating model performance and visualizing trained
filters. When run as __main__, evaluates the currently saved model on a test
set.
"""

# Standard imports
from __future__ import annotations
from pathlib import Path
import os
import sys
import getopt

# Third Party Imports
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Local Imports
from data import CheckboxData, ImageLike, PathLike, Prediction
from model import CheckboxCNNv1, CheckboxCNNv2

# Typing
from typing import (
    Callable,
    Optional,
    Union,
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


def infer(model: torch.nn.Module, image: Union[ImageLike, PathLike]) -> Prediction:
    """
    Given a model and an image, returns the model's prediction of the class

     Parameters:
        model:
            The model to use as the predictor

        image:
            a 3 dimensional array or path to an image file, whose class will
            be inferred by the model. Images must be 3 channel (RGB).

    Returns:
        The class of the image as a string
    """
    with torch.no_grad():
        model.eval()
        data = CheckboxData()
        data.load()

        if isinstance(image, (Path, str)):
            image = data.transform(cv2.imread(str(image)))

        # model only accepts batches, need a 'batch' of size 1
        image = torch.unsqueeze(image, 0)
        pred = model(image).argmax().item()
        return data.pred_to_label[pred]


if __name__ == "__main__":
    weights = Path.joinpath(
        Path(os.path.abspath(__file__)).parents[1], "weights/83v2.bin"
    )
    model = CheckboxCNNv2(weights)
    path = Path.joinpath(
        Path(os.path.abspath(__file__)).parents[1],
        "data/checked/0a4cbf5a03dd31a4782e752cf1fbd5d6.png",
    )
    print(infer(model, path))
