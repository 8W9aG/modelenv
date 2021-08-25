"""The implementation of a pytorch model."""
import enum
import typing
import sys

import numpy as np
import torch.nn as nn

from .model import Model
from .constants import ACTION_PARAMETER_COUNT, LAYER_PARAMETER_COUNT
from .norm import normalise, normalise_bool, denormalise, denormalise_bool


MAX_LAYER_TYPES = 200
MAX_CHANNEL_MODIFIERS = 10
MAX_KERNEL_SIZE_MODIFIERS = 20
MAX_STRIDE_MODIFIERS = MAX_KERNEL_SIZE_MODIFIERS
MAX_PADDING_MODIFIERS = MAX_STRIDE_MODIFIERS
MAX_PADDING_MODE_MODIFIERS = MAX_PADDING_MODIFIERS
MAX_DILATION_MODIFIERS = MAX_PADDING_MODIFIERS
MAX_GROUP_MODIFIERS = 5
PADDING_VALID = "valid"
PADDING_SAME = "same"
PADDING_MODE_ZEROS = "zeros"
PADDING_MODE_REFLECT = "reflect"
PADDING_MODE_REPLICATE = "replicate"
PADDING_MODE_CIRCULAR = "circular"


class TorchLayer(enum.IntEnum):
    """The frameworks for generating a model."""
    Conv1D = 0

class ChannelModifiers(enum.IntEnum):
    """The modifiers to use on channels to a module."""
    Quarter = 0
    Half = 1
    Whole = 2
    Double = 3
    Quadruple = 4

class PaddingModifiers(enum.IntEnum):
    """The modifiers to use on paddings"""
    Valid = 0
    Same = 1

class PaddingModeModifiers(enum.IntEnum):
    """The modifiers to use on padding modes"""
    Zeros = 0
    Reflect = 1
    Replicate = 2
    Circular = 3

def modify_channel(channels: int, channel_modifier: int) -> int:
    """Modify the channel"""
    if channel_modifier == ChannelModifiers.Quarter:
        return channels / 4
    if channel_modifier == ChannelModifiers.Half:
        return channels / 2
    if channel_modifier == ChannelModifiers.Whole:
        return channels
    if channel_modifier == ChannelModifiers.Double:
        return channels * 2
    if channel_modifier == ChannelModifiers.Quadruple:
        return channels * 4

def channel_modification(input_channels: int, output_channels: int) -> ChannelModifiers:
    """Perform a channel modification."""
    modifiers = {
        ChannelModifiers.Quarter: 0.25,
        ChannelModifiers.Half: 0.5,
        ChannelModifiers.Whole: 1.0,
        ChannelModifiers.Double: 2.0,
        ChannelModifiers.Quadruple: 4.0,
    }
    modification = float(output_channels) / float(input_channels)
    best_modification = None
    best_modification_diff = sys.float_info.max
    for modifier in modifiers:
        diff = abs(modification - modifiers[modifier])
        if diff < best_modification_diff:
            best_modification_diff = diff
            best_modification = modifier
    return best_modification

def padding_modifier_to_str(padding_modifier: int) -> typing.Optional[str]:
    """Convert the padding modifiers to strings."""
    if padding_modifier == PaddingModifiers.Valid:
        return PADDING_VALID
    if padding_modifier == PaddingModifiers.Same:
        return PADDING_SAME
    return None

def padding_str_to_modifier(padding: str) -> PaddingModifiers:
    """Convert the padding string to a padding modifier"""
    if padding == PADDING_VALID:
        return PaddingModifiers.Valid
    if padding == PADDING_SAME:
        return PaddingModifiers.Same
    raise Exception("No valid padding str modifier")

def padding_mode_modifier_to_str(padding_mode_modifier: int) -> typing.Optional[str]:
    """Convert the padding mode modifier to a string."""
    if padding_mode_modifier == PaddingModeModifiers.Zeros:
        return PADDING_MODE_ZEROS
    if padding_mode_modifier == PaddingModeModifiers.Reflect:
        return PADDING_MODE_REFLECT
    if padding_mode_modifier == PaddingModeModifiers.Replicate:
        return PADDING_MODE_REPLICATE
    if padding_mode_modifier == PaddingModeModifiers.Circular:
        return PADDING_MODE_CIRCULAR
    return None

def padding_mode_str_to_modifier(padding_mode: str) -> PaddingModeModifiers:
    """Conver thte padding mode string to a padding mode modifier"""
    if padding_mode == PADDING_MODE_ZEROS:
        return PaddingModeModifiers.Zeros
    if padding_mode == PADDING_MODE_REFLECT:
        return PaddingModeModifiers.Reflect
    if padding_mode == PADDING_MODE_REPLICATE:
        return PaddingModeModifiers.Replicate
    if padding_mode == PADDING_MODE_CIRCULAR:
        return PaddingModeModifiers.Circular
    raise Exception("No valid padding mode str modifier")

class TorchModel(Model):
    """A model backed by pytorch."""
    def __init__(self, action: np.array, example_data: np.array, example_output: np.array) -> None:
        super(TorchModel, self).__init__(example_data, example_output)
        self.network = nn.Sequential()
        self.add_layer(action)

    def add_layer(self, action: np.array):
        layer = action[ACTION_PARAMETER_COUNT:]
        layer_type = normalise(layer[0], MAX_LAYER_TYPES)
        if self.network:
            del self.network[-1]
        if layer_type == TorchLayer.Conv1D:
            self.network.add_module(
                "conv1d-" + str(len(self.network) + 1),
                nn.LazyConv1d(
                    modify_channel(len(self.example_output), normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                    stride=normalise(layer[3], MAX_STRIDE_MODIFIERS),
                    padding=padding_modifier_to_str(normalise(layer[4], MAX_PADDING_MODIFIERS)),
                    padding_mode=padding_mode_modifier_to_str(normalise(layer[5], MAX_PADDING_MODE_MODIFIERS)),
                    dilation=normalise(layer[6], MAX_DILATION_MODIFIERS),
                    groups=normalise(layer[7], MAX_GROUP_MODIFIERS),
                    bias=normalise_bool(layer[8]),
                )
            )
        # Add a linear layer to the end to force it to conform
        self.network.add_module("linear-end", nn.LazyLinear(len(self.example_output)))

    def remove_layer(self):
        if not self.network:
            return
        del self.network[-1]

    def state(self) -> np.array:
        network_state = []
        output = len(self.example_data)
        for module in self.network:
            layer_state = [0.0 for _ in range(LAYER_PARAMETER_COUNT)]
            if isinstance(module, nn.LazyConv1d):
                layer_state[0] = denormalise(TorchLayer.Conv1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
                layer_state[3] = denormalise(module.stride, MAX_STRIDE_MODIFIERS)
                layer_state[4] = denormalise(padding_str_to_modifier(module.padding), MAX_PADDING_MODIFIERS)
                layer_state[5] = denormalise(padding_mode_str_to_modifier(module.padding_mode), MAX_PADDING_MODE_MODIFIERS)
                layer_state[6] = denormalise(module.dilation, MAX_DILATION_MODIFIERS)
                layer_state[7] = denormalise(module.groups, MAX_GROUP_MODIFIERS)
                layer_state[8] = denormalise_bool(module.bias)
            network_state.extend(layer_state)
        return np.array(network_state)
