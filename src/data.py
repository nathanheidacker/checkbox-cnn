# -*- coding: utf-8 -*-
"""
Helper functions for loading train, test, and validation split
"""

# Standard Imports
from __future__ import annotations
from pathlib import Path
import os

# Third Party Imports
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Typing
from typing import (
    Optional,
    Union,
)

PathLike = Union[Path, str]


class CheckboxData:
    """
    Class responsible for loading image data for CNN training

    Parameters:
        path: Optional parameter specifying the data's root directory

    Attributes:
        path (Path):
            A Path object representing the root directory of the data

        image_paths (dict[str, list[str]]):
            A dictionary representing the file structure of the local data
    """

    def __init__(self, path: Optional[PathLike] = None) -> None:
        self.path = Path.joinpath(
            Path(os.path.abspath(__file__)).parents[1], (path or "data")
        )

    def _get_dataset(self) -> ImageFolder:
        """
        Returns the full, unsplit dataset
        """

        def square_pad(image_tensor: torch.Tensor) -> torch.Tensor:
            """
            Returns an image tensor padded such that width and height are equal
            """
            max_lw = max(image_tensor.shape[1:])
            horizontal = (max_lw - image_tensor.shape[1]) // 2
            vertical = (max_lw - image_tensor.shape[2]) // 2
            padding = (vertical, horizontal)
            return transforms.functional.pad(
                image_tensor, padding, fill=0, padding_mode="edge"
            )

        transform = transforms.Compose(
            [transforms.ToTensor(), square_pad, transforms.Resize((512, 512))]
        )
        return ImageFolder(self.path, transform=transform)

    def _get_dataloaders(
        self, split: float = 0.2, batch_size: int = 32
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns dataloaders for a training and test split

        Parameters:
            split:
                Determines the proportion of the data that should be used in the
                validation split

            batch_size:
                Determines minibatch size for both loaders
        """
        data = self._get_dataset()
        n_test = int(len(data) * split)
        n_train = len(data) - n_test

        # Setting a manual seed for the generator to ensure that the validation
        # split is the same for different training runs (so long as the split
        # proportion is constant). Allows more accurate performance evaluation
        generator = torch.Generator().manual_seed(10)

        # Splitting the data
        train, test = random_split(data, (n_train, n_test), generator=generator)
        train = DataLoader(train, batch_size=batch_size, shuffle=True)
        test = DataLoader(test, batch_size=batch_size, shuffle=False)
        return train, test


if __name__ == "__main__":
    data = CheckboxData()
