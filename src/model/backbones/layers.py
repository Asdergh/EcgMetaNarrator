import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any, Optional


def get_activation(name: str, **kwargs: Any) -> nn.Module:
    registry: dict[str, Callable[..., nn.Module]] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "hardswish": nn.Hardswish,
        "hardsigmoid": nn.Hardsigmoid,
        "leaky_relu_p": lambda negative_slope=0.01, **kw: nn.LeakyReLU(negative_slope, **kw),
        "elu_alpha": lambda alpha=1.0, **kw: nn.ELU(alpha, **kw),
    }

    key = name.lower().strip()
    if key not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown activation '{name}'. Available: {available}"
        )
    factory = registry[key]
    try:
        return factory(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments for activation '{name}': {kwargs}. "
            f"Original error: {e}"
        ) from e
    
class Film(nn.Module):
    def __init__(self, features: int):
        self._alpha_gate = nn.Softmax(nn.Linear(features, 2), nn.Sigmoid())
        self._beta = nn.Softmax(nn.Linear(features, 1), nn.GELU())

    def forward(self, x: th.Tensor):
        alpha, gamma = self._alpha_gate(x).chunk(2, dim=-1)
        beta = self._beta(x)
        x_shifted = alpha*x + beta
        x_film = gamma*x + (1 - gamma)*x_shifted
        return x_film
    
# class ConvStack(nn.Module):
#     def __init__(self, 
#                 in_chanels: int,
#                 out_channels: Optional[int]=None,
#                 film_filtration: bool=False,
#                 activation: str="relu",
#                 depth: int=3):
#         """ConvStack with wodnsampling maxpool2d at the end of a block. """
#         out_channels = out_channels if out_channels is not None else in_chanels
#         self._conv_stack = [nn.Conv2d(in_channels=in_chanels, 
#                                     out_channels=out_channels, 
#                                     3, 1, 1, 1)]