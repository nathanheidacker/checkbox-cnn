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
from .data import CheckboxData

# Typing
from typing import Callable


def evaluate(model: nn.Module, data: DataLoader, criterion: Callable = None) -> str:
    pass


def visualize():
    pass


if __name__ == "__main__":
    data = CheckboxData()
    train, test = data.load()
