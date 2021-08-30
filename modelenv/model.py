"""The generic class representing a model."""
import enum

import numpy as np


class ModelFramework(enum.IntEnum):
    """The frameworks for generating a model."""
    Torch = 0
    Tensorflow = 1
    Scipy = 2
    XGBoost = 3


class Model:
    """The abstraction for representing a model on any framework."""
    def __init__(self, example_data: np.array, example_output: np.array) -> None:
        self.example_data = example_data
        self.example_output = example_output

    def add_layer(self, action: np.array):
        """Add a layer to the model."""
        raise Exception("add_layer not implemented")

    def remove_layer(self):
        """Removes a layer from the model."""
        raise Exception("remove_layer not implemented")

    def state(self) -> np.array:
        """Finds the current observation state of the model."""
        raise Exception("state not implemented")

    def train(self) -> float:
        """Trains the current model."""
        return 0.0

    def print(self):
        """Prints out the current model architecture."""
        raise Exception("print not implemented")
