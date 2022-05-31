# -*- coding: utf-8 -*-
"""
The model for classifying checked boxes
"""

# Standard Imports
from __future__ import annotations

# Third Party Imports
import torch

# Local Imports
from .data import PathLike

# Typing
from typing import (
    Optional
)



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
        pass

    def forward(self, image_tensor: torch.Tensor) -> 