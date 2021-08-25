"""Normalise ranges."""
import math


def normalise(input: float, classes: int, low: float = -1.0, high: float = 1.0) -> int:
    """Normalise an input."""
    total = abs(low - high)
    return int(math.floor(((input + (total / 2.0)) / total) * classes))

def normalise_bool(input: float, low: float = -1.0, high: float = 1.0) -> bool:
    """Normalise an input to a boolean."""
    return normalise(input, 2, low=low, high=high) == 1

def denormalise(input: int, classes: int, low: float = -1.0, high: float = 1.0) -> float:
    """Denormalise the input to a range."""
    total = abs(low - high)
    return ((float(input) / float(classes)) * total) - (total / 2.0)

def denormalise_bool(input: bool, low: float = -1.0, high: float = 1.0) -> float:
    """Denormalise the input to a bool."""
    return denormalise(int(input), 2, low=low, high=high)
