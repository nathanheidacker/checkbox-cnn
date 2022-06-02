# -*- coding: utf-8 -*-
"""
Helper functions for loading train, test, and validation split
"""

# Standard Imports
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import os

# Third Party Imports
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Typing
from typing import (
    Optional,
    Union,
    Iterable
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

    def _expand_dataset(self, image_folder: ImageFolder, n_samples: int) -> TensorDataset:
        """
        Expands an ImageFolder dataset using random transforms such that all
        classes have n_samples

        Parameters:
            image_folder:
                The ImageFolder dataset to expand from

            n_samples:
                The number of samples for the

        Returns:
            A TensorDataset containing the expanded data of length n_samples *
            n_classes
        """
        samples = None
        if isinstance(image_folder, Subset):
            samples = [image_folder.dataset.samples[i] for i in image_folder.indices]
        else:
            samples = image_folder.samples
        class_counts = Counter([sample[1] for sample in samples])
        if n_samples <= sum(class_counts.values()) / len(class_counts): return image_folder
        random_transform = transforms.Compose([
            transforms.RandomRotation((-45,45)),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.7)
        ])
        X, y = [], []
        originals = defaultdict(list)
        for sample, label in image_folder:
            originals[label].append(sample)
            X.append(sample)
            y.append(label)

        for k, v in class_counts.items():
            i = 0
            end = len(originals[k]) - 1
            while v < n_samples:
                img = originals[k][i]
                X.append(img)
                y.append(k)

                v += 1
                if i == end:
                    i = 0
                else:
                    i += 1

        X, y = torch.stack(X), torch.LongTensor(y)
        return TensorDataset(X, y)

    def _get_dataloaders(
        self, data: Dataset, n_samples: int, split: float, batch_size: int
    ) -> tuple[DataLoader, DataLoader]:
        """
        Returns dataloaders for a training and test split
        """
        n_test = int(len(data) * split)
        n_train = len(data) - n_test

        # Setting a manual seed for the generator to ensure that the validation
        # split is the same for different training runs (so long as the split
        # proportion is constant). Allows more accurate performance evaluation
        generator = torch.Generator().manual_seed(10)

        # Splitting the data
        train, test = random_split(data, (n_train, n_test), generator=generator)
        train = self._expand_dataset(train, n_samples)
        train = DataLoader(train, batch_size=batch_size, shuffle=True)
        test = DataLoader(test, batch_size=batch_size, shuffle=False)
        return train, test

    def load(self, n_samples: int = 200, split: int = 0.2, batch_size: int = 16) -> tuple(Iterable, Iterable):
        """
        Returns dataloaders for a training and test split

        Parameters:
            n_samples:
                The number of samples of each class in the training set

            split:
                Determines the proportion of the original data that should be 
                used in the validation split

            batch_size:
                Determines minibatch size for both loaders
        """
        dataset = self._get_dataset()
        return self._get_dataloaders(dataset, n_samples, split, batch_size)


if __name__ == "__main__":
    # Instantiating the data
    data = CheckboxData()
    train, test = data.load()

    # Stats about the training set
    batches = len(train)
    shape = next(iter(train))[0].shape
    classes = Counter(sample[1].item() for sample in train.dataset)
    train_stats = (
        f"BATCHES: {batches} batches per epoch\n"
        f"BATCH SIZE: {shape[0]} samples per batch\n"
        f"IMAGES: {shape[1]} channel images of size {[x for x in shape[2:]]}\n"
        f"SAMPLES: {len(train.dataset)} total samples\n"
        f"CLASS BREAKDOWN: {classes}"
    )

    # Stats about the validation set
    batches = len(test)
    shape = next(iter(test))[0].shape
    classes = Counter(sample[1] for sample in test.dataset)
    test_stats = (
        f"BATCHES: {batches} batches per epoch\n"
        f"BATCH SIZE: {shape[0]} samples per batch\n"
        f"IMAGES: {shape[1]} channel images of size {[x for x in shape[2:]]}\n"
        f"SAMPLES: {len(test.dataset)} total samples\n"
        f"CLASS BREAKDOWN: {classes}"
    )

    print(
        f"TRAINING STATS\n--------------\n{train_stats}\n\n"
        f"TEST STATS\n----------\n{test_stats}"
    )