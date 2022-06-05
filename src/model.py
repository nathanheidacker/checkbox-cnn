# -*- coding: utf-8 -*-
"""
The model for classifying checked boxes
"""

# Standard Imports
from __future__ import annotations
from pathlib import Path
import os
import io
import sys

# Third Party Imports
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Local Imports
from data import CheckboxData, PathLike, ImageLike

# Typing
from typing import Optional, Union


class _CheckboxCNN(torch.nn.Module):
    """
    Base class for CheckboxCNN models that provides weight initialization and
    model visualization functionality
    """

    def _init_weights(self, weights: Optional[PathLike] = None) -> None:
        if weights:
            loaded = None
            if torch.cuda.is_available():
                loaded = torch.load(weights, map_location=torch.device("cuda"))
            else:
                loaded = torch.load(weights, map_location=torch.device("cpu"))
            self.load_state_dict(loaded)

        else:
            # Conv2D layers use xavier normal initialization
            for layer in self.conv.children():
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                elif isinstance(layer, nn.BatchNorm2d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()

            # Linear layers use xavier uniform
            for layer in self.classifier.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                elif isinstance(layer, nn.BatchNorm1d):
                    layer.weight.data.fill_(1)
                    layer.bias.data.zero_()

    @staticmethod
    def _fig_to_np(fig: plt.figure) -> np.ndarray:
        """
        Given a matplotlib figure, returns an np array of the image
        """
        with io.BytesIO() as buff:
            fig.savefig(buff, format="raw")
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        return data.reshape((int(h), int(w), -1))

    @staticmethod
    def _resize_fig(img: np.ndarray) -> plt.figure:
        """
        Given a numpy array, returns a plt figure with canvas size matching 
        the array shape
        """

        # get the dimensions
        height, width, channels = img.shape

        # get the size in inches
        dpi = 30
        width /= dpi
        height /= dpi

        # plot and save in the same size as the original
        fig = plt.figure(figsize=(width, height))

        ax = plt.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])
        ax.imshow(img, interpolation="none")
        return fig

    def visualize(
        self,
        layer: int,
        out: Union[PathLike, type],
        image: Union[ImageLike, PathLike, None] = None,
        show: bool = False,
        _transform: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """
        Visualizes a convolutional layer of the network with a matplotlib
        output. Given an image, visualizes the image at [layer] stage of the
        forward pass

        Parameters:
            layer:
                The layer to be visualized

            out:
                The output path of the image

            image:
                A 3D array or filepath to an image to be visualized instead of the layer's filters themselves

            show:
                Whether or not the image should be displayed in addition to
                being saved
        """
        self.eval()

        # Loading transforms
        if _transform is None:
            data = CheckboxData()
            data.load(1)
            _transform = data.transform

        # Ensure the layer is valid
        layer = (layer - 1) * 4
        modules = [x for x in self.conv[:]]
        if not layer in range(len(modules)):
            raise ValueError(
                f"Not a valid convolutional layer, please select an integer value between 1 and {len(modules) / 4:.0f}"
            )

        # Visualizing an image forward pass
        if image is not None:
            with torch.no_grad():
                if isinstance(image, (Path, str)):
                    image = cv2.imread(str(image))
                image = _transform(image).unsqueeze(0)
                modules = nn.Sequential(*modules[: layer + 1])
                forward = modules(image)[0]
                n = int((forward.shape[0] ** 0.5) // 1)
                fig, axes = plt.subplots(nrows=n, ncols=n)
                for i, row in enumerate(axes):
                    for j, ax in enumerate(row):
                        ax.imshow(forward[i * 4 + j])
                        ax.set_axis_off()

                # Visualization formatting
                fig.suptitle(f"Image forward pass at conv layer {layer / 4 + 1:.0f}")
                fig.tight_layout()

        # Visualizing the maximum activation of the filter
        else:
            layer = self.conv[layer]
            print(layer.weight.shape)
            raise NotImplementedError("This functionality has not yet been implemented")

        # Outputting to path
        if isinstance(out, (Path, str)):
            fig.savefig(out)

        # Closing the figure if not shown
        result = self._fig_to_np(fig)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return result

    def report(self, out: PathLike, image: Union[ImageLike, PathLike]) -> None:
        """
        Creates a comprehensive visualization of this model's weights for each
        convolutional layer, outputting the final report image to out

        Parameters:
            out:
                The output path of the final report

            image:
                The image to be used in the report
        """

        # Getting the transform
        data = CheckboxData()
        data.load(1)
        transform = data.transform

        # Getting all relevant visualizations
        n_layers = (len([x for x in self.conv]) // 4) + 1
        """
        map_vizs = [
            self.visualize(i, np.ndarray, show=False, _transform=transform)
            for i in range(1, n_layers)
        ]
        """
        image_vizs = [
            self.visualize(i, np.ndarray, image, False, transform)
            for i in range(1, n_layers)
        ]

        # Combining into a single image
        # map_viz = np.concatenate(map_vizs, axis=0)
        image_viz = np.concatenate(image_vizs, axis=0)
        # viz = np.concatenate((map_viz, image_viz), axis=1)

        # Getting the resized image file, saving
        fig = self._resize_fig(image_viz)
        fig.savefig(out)


class CheckboxCNNv1(_CheckboxCNN):
    """
    A shallow convolutional neural network designed to detect and classify the
    state of HTML checkboxes

    Parameters:
        weights:
            An optional parameter specifying a path to trained weights for this
            model
    """

    def __init__(self, weights: Optional[PathLike] = None) -> None:
        super().__init__()

        # The convolutional forward pass
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification of convolved features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(131072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 3),
        )

        # Loading/initializing weights
        self._init_weights(weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(image))


class CheckboxCNNv2(_CheckboxCNN):
    """
    A deep convolutional neural network designed to detect and classify the
    state of HTML checkboxes

    Parameters:
        weights:
            An optional parameter specifying a path to trained weights for this
            model
    """

    def __init__(self, weights: Optional[PathLike] = None) -> None:
        super().__init__()

        # The convolutional forward pass
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification of convolved features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 3),
        )

        # Loading/initializing weights
        self._init_weights(weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(image))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = sys.argv
    if len(args) < 2:
        raise RuntimeError(f"Please specify the model to visualize")
    elif len(args) > 2:
        raise RuntimeError(f"Unrecognized arguments {args[2:]}")
    elif args[1] not in ["v1", "v2"]:
        raise RuntimeError(f"Please select 'v1' or 'v2' for the model version")
    version = args[1]
    image_path = Path.joinpath(
        Path(os.path.abspath(__file__)).parents[1],
        "data/checked/4dd6ca2261e45d2b3abed5d84b55654d.png",
    )

    if version == "v1":
        weights = Path.joinpath(
            Path(os.path.abspath(__file__)).parents[1], "weights/81v1.bin"
        )
        model = CheckboxCNNv1(weights)
        model.report("v1report.png", image_path)
    elif version == "v2":
        weights = Path.joinpath(
            Path(os.path.abspath(__file__)).parents[1], "weights/83v2.bin"
        )
        model = CheckboxCNNv2(weights)
        model.report("v2report.png", image_path)
