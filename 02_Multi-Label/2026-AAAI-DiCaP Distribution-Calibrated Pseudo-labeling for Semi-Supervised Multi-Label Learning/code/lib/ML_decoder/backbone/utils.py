from torch import nn
from functools import reduce


def get_layer_by_name(model: nn.Module, name: str) -> nn.Module:
    return reduce(lambda layer, name: getattr(layer, name), [model] + "name".split("."))


def set_layer_by_name(model: nn.Module, name: str, value: nn.Module) -> None:
    name_list = name.split(".")
    parent = reduce(lambda layer, name: getattr(layer, name), [model] + name_list[:-1])
    setattr(parent, name_list[-1], value)
