"""The implementation of a pytorch model."""
import enum
import sys

import numpy as np
import torch.nn as nn

from .model import Model
from .constants import ACTION_PARAMETER_COUNT, LAYER_PARAMETER_COUNT
from .norm import normalise, denormalise


MAX_LAYER_TYPES = 200
MAX_CHANNEL_MODIFIERS = 10
MAX_KERNEL_SIZE_MODIFIERS = 20
MAX_NORM_TYPE_MODIFIERS = 2
MAX_PADDING_MODIFIERS = MAX_KERNEL_SIZE_MODIFIERS
MAX_NUM_HEAD_MODIFIERS = 10
MAX_NUM_CLASSES = 10


class TorchLayer(enum.IntEnum):
    """The layers for generating a model."""
    Conv1D = 0
    Conv2D = 1
    Conv3D = 2
    ConvTranspose1D = 3
    ConvTranspose2D = 4
    ConvTranspose3D = 5
    Unfold = 6
    Fold = 7
    MaxPool1D = 8
    MaxPool2D = 9
    MaxPool3D = 10
    MaxUnpool1D = 11
    MaxUnpool2D = 12
    MaxUnpool3D = 13
    AvgPool1D = 14
    AvgPool2D = 15
    AvgPool3D = 16
    FractionalMaxPool2D = 17
    FractionalMaxPool3D = 18
    LPPool1D = 19
    LPPool2D = 20
    AdaptiveMaxPool1D = 21
    AdaptiveMaxPool2D = 22
    AdaptiveMaxPool3D = 23
    AdaptiveAvgPool1D = 24
    AdaptiveAvgPool2D = 25
    AdaptiveAvgPool3D = 26
    ReflectionPad1D = 27
    ReflectionPad2D = 28
    ReplicationPad1D = 29
    ReplicationPad2D = 30
    ReplicationPad3D = 31
    ZeroPad2D = 32
    ConstantPad1D = 33
    ConstantPad2D = 34
    ConstantPad3D = 35
    ELU = 36
    HardShrink = 37
    HardSigmoid = 38
    HardTanh = 39
    HardSwish = 40
    LeakyReLU = 41
    LogSigmoid = 42
    MultiHeadAttention = 43
    PReLU = 44
    ReLU = 45
    ReLU6 = 46
    RReLU = 47
    SELU = 48
    CELU = 49
    GELU = 50
    Sigmoid = 51
    SiLU = 52
    Mish = 53
    Softplus = 54
    Softshrink = 55
    Softsign = 56
    Tanh = 57
    Tanhshrink = 58
    Threshold = 59
    Softmin = 60
    Softmax = 61
    Softmax2D = 62
    LogSoftmax = 63
    AdaptiveLogSoftmaxWithLoss = 64

class ChannelModifiers(enum.IntEnum):
    """The modifiers to use on channels to a module."""
    Quarter = 0
    Half = 1
    Whole = 2
    Double = 3
    Quadruple = 4

class NumHeads(enum.IntEnum):
    """The number of heads to use on an attention module."""
    TwoHeads = 0
    FourHeads = 1
    SixHeads = 2
    EightHeads = 3
    TenHeads = 4

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

def num_head_modification(head_modifier: int) -> int:
    """Perform the modification on num_heads."""
    if head_modifier == NumHeads.TwoHeads:
        return 2
    if head_modifier == NumHeads.FourHeads:
        return 4
    if head_modifier == NumHeads.SixHeads:
        return 6
    if head_modifier == NumHeads.EightHeads:
        return 8
    if head_modifier == NumHeads.TenHeads:
        return 10

def denormalise_head_modification(num_heads: int) -> NumHeads:
    if num_heads == 2:
        return NumHeads.TwoHeads
    if num_heads == 4:
        return NumHeads.FourHeads
    if num_heads == 6:
        return NumHeads.SixHeads
    if num_heads == 8:
        return NumHeads.EightHeads
    if num_heads == 10:
        return NumHeads.TenHeads

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
        input_channels = len(self.example_data)
        if self.network:
            input_channels = self.network[-1].in_channels
        if layer_type == TorchLayer.Conv1D:
            self.network.add_module(
                "conv1d-" + str(len(self.network) + 1),
                nn.LazyConv1d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.Conv2D:
            self.network.add_module(
                "conv2d-" + str(len(self.network) + 1),
                nn.LazyConv2d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.Conv3D:
            self.network.add_module(
                "conv3d-" + str(len(self.network) + 1),
                nn.LazyConv3d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ConvTranspose1D:
            self.network.add_module(
                "convtranspose1d-" + str(len(self.network) + 1),
                nn.LazyConvTranspose1d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ConvTranspose2D:
            self.network.add_module(
                "convtranspose2d-" + str(len(self.network) + 1),
                nn.LazyConvTranspose2d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ConvTranspose3D:
            self.network.add_module(
                "convtranspose3d-" + str(len(self.network) + 1),
                nn.LazyConvTranspose3d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.Unfold:
            self.network.add_module(
                "unfold-" + str(len(self.network) + 1),
                nn.Unfold(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.Fold:
            self.network.add_module(
                "fold-" + str(len(self.network) + 1),
                nn.Fold(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxPool1D:
            self.network.add_module(
                "maxpool1d-" + str(len(self.network) + 1),
                nn.MaxPool1d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxPool2D:
            self.network.add_module(
                "maxpool2d-" + str(len(self.network) + 1),
                nn.MaxPool2d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxPool3D:
            self.network.add_module(
                "maxpool3d-" + str(len(self.network) + 1),
                nn.MaxPool3d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxUnpool1D:
            self.network.add_module(
                "maxunpool1d-" + str(len(self.network) + 1),
                nn.MaxUnpool1d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxUnpool2D:
            self.network.add_module(
                "maxunpool2d-" + str(len(self.network) + 1),
                nn.MaxUnpool2d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.MaxUnpool3D:
            self.network.add_module(
                "maxunpool3d-" + str(len(self.network) + 1),
                nn.MaxUnpool3d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.AvgPool1D:
            self.network.add_module(
                "avgpool1d-" + str(len(self.network) + 1),
                nn.AvgPool1d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.AvgPool2D:
            self.network.add_module(
                "avgpool2d-" + str(len(self.network) + 1),
                nn.AvgPool2d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.AvgPool3D:
            self.network.add_module(
                "avgpool3d-" + str(len(self.network) + 1),
                nn.AvgPool3d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.FractionalMaxPool2D:
            self.network.add_module(
                "fractionalmaxpool2d-" + str(len(self.network) + 1),
                nn.FractionalMaxPool2d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.FractionalMaxPool3D:
            self.network.add_module(
                "fractionalmaxpool3d-" + str(len(self.network) + 1),
                nn.FractionalMaxPool3d(
                    normalise(layer[1], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.LPPool1D:
            self.network.add_module(
                "lppool1d-" + str(len(self.network) + 1),
                nn.LPPool1d(
                    normalise(layer[1], MAX_NORM_TYPE_MODIFIERS) + 1,
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.LPPool2D:
            self.network.add_module(
                "lppool2d-" + str(len(self.network) + 1),
                nn.LPPool2d(
                    normalise(layer[1], MAX_NORM_TYPE_MODIFIERS) + 1,
                    normalise(layer[2], MAX_KERNEL_SIZE_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.AdaptiveMaxPool1D:
            self.network.add_module(
                "adaptivemaxpool1d-" + str(len(self.network) + 1),
                nn.AdaptiveMaxPool1d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.AdaptiveMaxPool2D:
            self.network.add_module(
                "adaptivemaxpool2d-" + str(len(self.network) + 1),
                nn.AdaptiveMaxPool2d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.AdaptiveMaxPool3D:
            self.network.add_module(
                "adaptivemaxpool3d-" + str(len(self.network) + 1),
                nn.AdaptiveMaxPool3d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.AdaptiveAvgPool1D:
            self.network.add_module(
                "adaptiveavgpool1d-" + str(len(self.network) + 1),
                nn.AdaptiveAvgPool1d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.AdaptiveAvgPool2D:
            self.network.add_module(
                "adaptiveavgpool2d-" + str(len(self.network) + 1),
                nn.AdaptiveAvgPool2d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.AdaptiveAvgPool3D:
            self.network.add_module(
                "adaptiveavgpool3d-" + str(len(self.network) + 1),
                nn.AdaptiveAvgPool3d(
                    modify_channel(input_channels, normalise(layer[1], MAX_CHANNEL_MODIFIERS)),
                )
            )
        elif layer_type == TorchLayer.ReflectionPad1D:
            self.network.add_module(
                "reflectionpad1d-" + str(len(self.network) + 1),
                nn.ReflectionPad1d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ReflectionPad2D:
            self.network.add_module(
                "reflectionpad2d-" + str(len(self.network) + 1),
                nn.ReflectionPad2d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ReplicationPad1D:
            self.network.add_module(
                "replicationpad1d-" + str(len(self.network) + 1),
                nn.ReplicationPad1d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ReplicationPad2D:
            self.network.add_module(
                "replicationpad2d-" + str(len(self.network) + 1),
                nn.ReplicationPad2d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ReplicationPad3D:
            self.network.add_module(
                "replicationpad3d-" + str(len(self.network) + 1),
                nn.ReplicationPad3d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ZeroPad2D:
            self.network.add_module(
                "zeropad2d-" + str(len(self.network) + 1),
                nn.ZeroPad2d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                )
            )
        elif layer_type == TorchLayer.ConstantPad1D:
            self.network.add_module(
                "constantpad1d-" + str(len(self.network) + 1),
                nn.ConstantPad1d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                    layer[2],
                )
            )
        elif layer_type == TorchLayer.ConstantPad2D:
            self.network.add_module(
                "constantpad2d-" + str(len(self.network) + 1),
                nn.ConstantPad2d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                    layer[2],
                )
            )
        elif layer_type == TorchLayer.ConstantPad3D:
            self.network.add_module(
                "constantpad3d-" + str(len(self.network) + 1),
                nn.ConstantPad3d(
                    normalise(layer[1], MAX_PADDING_MODIFIERS) + 1,
                    layer[2],
                )
            )
        elif layer_type == TorchLayer.ELU:
            self.network.add_module(
                "elu-" + str(len(self.network) + 1),
                nn.ELU()
            )
        elif layer_type == TorchLayer.HardShrink:
            self.network.add_module(
                "hardshrink-" + str(len(self.network) + 1),
                nn.Hardshrink()
            )
        elif layer_type == TorchLayer.HardSigmoid:
            self.network.add_module(
                "hardsigmoid-" + str(len(self.network) + 1),
                nn.Hardsigmoid()
            )
        elif layer_type == TorchLayer.HardTanh:
            self.network.add_module(
                "hardtanh-" + str(len(self.network) + 1),
                nn.Hardtanh()
            )
        elif layer_type == TorchLayer.HardSwish:
            self.network.add_module(
                "hardswish-" + str(len(self.network) + 1),
                nn.Hardswish()
            )
        elif layer_type == TorchLayer.LeakyReLU:
            self.network.add_module(
                "leakyrelu-" + str(len(self.network) + 1),
                nn.LeakyReLU()
            )
        elif layer_type == TorchLayer.LogSigmoid:
            self.network.add_module(
                "logsigmoid-" + str(len(self.network) + 1),
                nn.LogSigmoid()
            )
        elif layer_type == TorchLayer.MultiHeadAttention:
            self.network.add_module(
                "multiheadattention-" + str(len(self.network) + 1),
                nn.MultiheadAttention(
                    input_channels,
                    num_head_modification(normalise(layer[1], MAX_NUM_HEAD_MODIFIERS))
                )
            )
        elif layer_type == TorchLayer.PReLU:
            self.network.add_module(
                "prelu-" + str(len(self.network) + 1),
                nn.PReLU()
            )
        elif layer_type == TorchLayer.ReLU:
            self.network.add_module(
                "relu-" + str(len(self.network) + 1),
                nn.ReLU()
            )
        elif layer_type == TorchLayer.ReLU6:
            self.network.add_module(
                "relu6-" + str(len(self.network) + 1),
                nn.ReLU6()
            )
        elif layer_type == TorchLayer.RReLU:
            self.network.add_module(
                "rrelu-" + str(len(self.network) + 1),
                nn.RReLU()
            )
        elif layer_type == TorchLayer.SELU:
            self.network.add_module(
                "selu-" + str(len(self.network) + 1),
                nn.SELU()
            )
        elif layer_type == TorchLayer.CELU:
            self.network.add_module(
                "celu-" + str(len(self.network) + 1),
                nn.CELU()
            )
        elif layer_type == TorchLayer.GELU:
            self.network.add_module(
                "gelu-" + str(len(self.network) + 1),
                nn.GELU()
            )
        elif layer_type == TorchLayer.Sigmoid:
            self.network.add_module(
                "sigmoid-" + str(len(self.network) + 1),
                nn.Sigmoid()
            )
        elif layer_type == TorchLayer.SiLU:
            self.network.add_module(
                "silu-" + str(len(self.network) + 1),
                nn.SiLU()
            )
        elif layer_type == TorchLayer.Mish:
            self.network.add_module(
                "mish-" + str(len(self.network) + 1),
                nn.Mish()
            )
        elif layer_type == TorchLayer.Softplus:
            self.network.add_module(
                "softplus-" + str(len(self.network) + 1),
                nn.Softplus()
            )
        elif layer_type == TorchLayer.Softshrink:
            self.network.add_module(
                "softshrink-" + str(len(self.network) + 1),
                nn.Softshrink()
            )
        elif layer_type == TorchLayer.Softsign:
            self.network.add_module(
                "softsign-" + str(len(self.network) + 1),
                nn.Softsign()
            )
        elif layer_type == TorchLayer.Tanh:
            self.network.add_module(
                "tanh-" + str(len(self.network) + 1),
                nn.Tanh()
            )
        elif layer_type == TorchLayer.Tanhshrink:
            self.network.add_module(
                "tanhshrink-" + str(len(self.network) + 1),
                nn.Tanhshrink()
            )
        elif layer_type == TorchLayer.Threshold:
            self.network.add_module(
                "threshold-" + str(len(self.network) + 1),
                nn.Threshold(layer[1], layer[2])
            )
        elif layer_type == TorchLayer.Softmin:
            self.network.add_module(
                "softmin-" + str(len(self.network) + 1),
                nn.Softmin()
            )
        elif layer_type == TorchLayer.Softmax:
            self.network.add_module(
                "softmax-" + str(len(self.network) + 1),
                nn.Softmax()
            )
        elif layer_type == TorchLayer.Softmax2D:
            self.network.add_module(
                "softmax2d-" + str(len(self.network) + 1),
                nn.Softmax2d()
            )
        elif layer_type == TorchLayer.LogSoftmax:
            self.network.add_module(
                "logsoftmax-" + str(len(self.network) + 1),
                nn.LogSoftmax()
            )
        elif layer_type == TorchLayer.AdaptiveLogSoftmaxWithLoss:
            self.network.add_module(
                "adaptivelogsoftmaxwithloss-" + str(len(self.network) + 1),
                nn.AdaptiveLogSoftmaxWithLoss(
                    input_channels,
                    normalise(layer[1], MAX_NUM_CLASSES),
                    [10],
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
            elif isinstance(module, nn.LazyConv2d):
                layer_state[0] = denormalise(TorchLayer.Conv2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.LazyConv3d):
                layer_state[0] = denormalise(TorchLayer.Conv3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.LazyConvTranspose1d):
                layer_state[0] = denormalise(TorchLayer.ConvTranspose1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.LazyConvTranspose2d):
                layer_state[0] = denormalise(TorchLayer.ConvTranspose2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.LazyConvTranspose3d):
                layer_state[0] = denormalise(TorchLayer.ConvTranspose3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.Unfold):
                layer_state[0] = denormalise(TorchLayer.Unfold, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.Fold):
                layer_state[0] = denormalise(TorchLayer.Fold, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(channel_modification(module.out_channels, output), MAX_CHANNEL_MODIFIERS)
                layer_state[2] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxPool1d):
                layer_state[0] = denormalise(TorchLayer.MaxPool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxPool2d):
                layer_state[0] = denormalise(TorchLayer.MaxPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxPool3d):
                layer_state[0] = denormalise(TorchLayer.MaxPool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxUnpool1d):
                layer_state[0] = denormalise(TorchLayer.MaxUnpool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxUnpool2d):
                layer_state[0] = denormalise(TorchLayer.MaxUnpool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.MaxUnpool3d):
                layer_state[0] = denormalise(TorchLayer.MaxUnpool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.AvgPool1d):
                layer_state[0] = denormalise(TorchLayer.AvgPool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.AvgPool2d):
                layer_state[0] = denormalise(TorchLayer.AvgPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.AvgPool3d):
                layer_state[0] = denormalise(TorchLayer.AvgPool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.FractionalMaxPool2d):
                layer_state[0] = denormalise(TorchLayer.FractionalMaxPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.FractionalMaxPool3d):
                layer_state[0] = denormalise(TorchLayer.FractionalMaxPool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS)
            elif isinstance(module, nn.LPPool1d):
                layer_state[0] = denormalise(TorchLayer.LPPool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.norm_type - 1, MAX_NORM_TYPE_MODIFIERS) - 1
            elif isinstance(module, nn.LPPool2d):
                layer_state[0] = denormalise(TorchLayer.LPPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.norm_type - 1, MAX_NORM_TYPE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveMaxPool1d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveMaxPool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveMaxPool2d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveMaxPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveMaxPool3d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveMaxPool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveAvgPool1d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveAvgPool1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveAvgPool2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.AdaptiveAvgPool3d):
                layer_state[0] = denormalise(TorchLayer.AdaptiveAvgPool3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.kernel_size - 1, MAX_KERNEL_SIZE_MODIFIERS) - 1
            elif isinstance(module, nn.ReflectionPad1d):
                layer_state[0] = denormalise(TorchLayer.ReflectionPad1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ReflectionPad2d):
                layer_state[0] = denormalise(TorchLayer.ReflectionPad2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ReplicationPad1d):
                layer_state[0] = denormalise(TorchLayer.ReplicationPad1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ReplicationPad2d):
                layer_state[0] = denormalise(TorchLayer.ReplicationPad2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ReplicationPad3d):
                layer_state[0] = denormalise(TorchLayer.ReplicationPad3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ZeroPad2d):
                layer_state[0] = denormalise(TorchLayer.ZeroPad2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
            elif isinstance(module, nn.ConstantPad1d):
                layer_state[0] = denormalise(TorchLayer.ConstantPad1D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
                layer_state[2] = module.value
            elif isinstance(module, nn.ConstantPad2d):
                layer_state[0] = denormalise(TorchLayer.ConstantPad2D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
                layer_state[2] = module.value
            elif isinstance(module, nn.ConstantPad3d):
                layer_state[0] = denormalise(TorchLayer.ConstantPad3D, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.padding - 1, MAX_PADDING_MODIFIERS) - 1
                layer_state[2] = module.value
            elif isinstance(module, nn.ELU):
                layer_state[0] = denormalise(TorchLayer.ELU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Hardshrink):
                layer_state[0] = denormalise(TorchLayer.HardShrink, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Hardsigmoid):
                layer_state[0] = denormalise(TorchLayer.HardSigmoid, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Hardtanh):
                layer_state[0] = denormalise(TorchLayer.HardTanh, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Hardswish):
                layer_state[0] = denormalise(TorchLayer.HardSwish, MAX_LAYER_TYPES)
            elif isinstance(module, nn.LeakyReLU):
                layer_state[0] = denormalise(TorchLayer.LeakyReLU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.LogSigmoid):
                layer_state[0] = denormalise(TorchLayer.LogSigmoid, MAX_LAYER_TYPES)
            elif isinstance(module, nn.MultiheadAttention):
                layer_state[0] = denormalise(TorchLayer.MultiHeadAttention, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(denormalise_head_modification(module.num_heads), MAX_NUM_HEAD_MODIFIERS)
            elif isinstance(module, nn.PReLU):
                layer_state[0] = denormalise(TorchLayer.PReLU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.ReLU):
                layer_state[0] = denormalise(TorchLayer.ReLU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.ReLU6):
                layer_state[0] = denormalise(TorchLayer.ReLU6, MAX_LAYER_TYPES)
            elif isinstance(module, nn.RReLU):
                layer_state[0] = denormalise(TorchLayer.RReLU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.SELU):
                layer_state[0] = denormalise(TorchLayer.SELU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.CELU):
                layer_state[0] = denormalise(TorchLayer.CELU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.GELU):
                layer_state[0] = denormalise(TorchLayer.GELU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Sigmoid):
                layer_state[0] = denormalise(TorchLayer.Sigmoid, MAX_LAYER_TYPES)
            elif isinstance(module, nn.SiLU):
                layer_state[0] = denormalise(TorchLayer.SiLU, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Mish):
                layer_state[0] = denormalise(TorchLayer.Mish, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Softplus):
                layer_state[0] = denormalise(TorchLayer.Softplus, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Softshrink):
                layer_state[0] = denormalise(TorchLayer.Softshrink, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Softsign):
                layer_state[0] = denormalise(TorchLayer.Softsign, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Tanh):
                layer_state[0] = denormalise(TorchLayer.Tanh, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Tanhshrink):
                layer_state[0] = denormalise(TorchLayer.Tanhshrink, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Threshold):
                layer_state[0] = denormalise(TorchLayer.Threshold, MAX_LAYER_TYPES)
                layer_state[1] = module.threshold
                layer_state[2] = module.value
            elif isinstance(module, nn.Softmin):
                layer_state[0] = denormalise(TorchLayer.Softmin, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Softmax):
                layer_state[0] = denormalise(TorchLayer.Softmax, MAX_LAYER_TYPES)
            elif isinstance(module, nn.Softmax2d):
                layer_state[0] = denormalise(TorchLayer.Softmax2D, MAX_LAYER_TYPES)
            elif isinstance(module, nn.LogSoftmax):
                layer_state[0] = denormalise(TorchLayer.LogSoftmax, MAX_LAYER_TYPES)
            elif isinstance(module, nn.AdaptiveLogSoftmaxWithLoss):
                layer_state[0] = denormalise(TorchLayer.AdaptiveLogSoftmaxWithLoss, MAX_LAYER_TYPES)
                layer_state[1] = denormalise(module.n_classes, MAX_NUM_CLASSES)
            network_state.extend(layer_state)
        return np.array(network_state)
