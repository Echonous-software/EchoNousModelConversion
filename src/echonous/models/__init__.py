from dataclasses import dataclass
from typing import Literal, TypeAlias, Callable

import torch.nn as nn


@dataclass
class ModuleInputOutput:
    name: str
    shape: tuple[int, ...]
    type: Literal['image'] | Literal['tensor'] = 'tensor'
    scale: float = 1.0


@dataclass
class Model:
    name: str
    version: str
    description: str
    module: nn.Module
    inputs: list[ModuleInputOutput]
    outputs: list[ModuleInputOutput]

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


ModelLoader: TypeAlias = Callable[[dict], Model]
