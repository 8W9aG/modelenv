"""A factory for creating models."""
import math

import numpy as np

from .model import Model, ModelFramework
from .torchmodel import TorchModel


def create_model(action: np.array) -> Model:
    """Create a model with an initial action."""
    model_framework = int(math.floor(((action[0] + 1.0) / 2.0) * (ModelFramework.Scipy + 1)))
    if model_framework == ModelFramework.Torch:
        return TorchModel(action)
    elif model_framework == ModelFramework.Tensorflow:
        pass
    elif model_framework == ModelFramework.Scipy:
        pass
