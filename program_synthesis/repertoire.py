from typing import List, Union
from fragile.backend import dtype, tensor, typing
import torch
from torch import nn

from program_synthesis.modules import Operator, OperatorParams


class BaseRepertoire(nn.Module):
    def __init__(
        self, functions: nn.ModuleList, n_registers: int,
    ):
        super(BaseRepertoire, self).__init__()
        self._registers = n_registers
        self._functions = functions

    def __len__(self):
        return self.n_functions

    @property
    def n_functions(self):
        return len(self._functions)

    def __getitem__(self, item: int):
        return self.functions[int(item)]

    @property
    def n_registers(self):
        return self._registers

    @property
    def seed(self):
        return self._seed

    @property
    def functions(self) -> torch.nn.ModuleList:
        return self._functions

    def forward_one_action(self, registers, action, grad: bool = True, **kwargs):
        if not grad:
            with torch.no_grad():
                return self[action](registers)
        return self[action](registers)

    def forward(
        self, registers, program: Union[List[int], typing.Tensor], grad: bool = True
    ) -> typing.Tensor:
        reshape = len(registers.shape) == 3
        if reshape:
            orig_shape = registers.shape
            registers = registers.reshape(-1, self.n_registers)
        for func in program:
            if func < 0:
                continue
            registers = self.forward_one_action(registers=registers, action=func, grad=grad)
        if reshape:
            registers = registers.reshape(orig_shape)
        return registers

    def get_empty_registers(
        self, n_walkers: int = 1, batch_size: int = 1, as_tensor: bool = False
    ) -> typing.Tensor:
        """Return an array of zeros representing a set of registers."""
        registers = tensor.zeros((n_walkers, batch_size, self.n_registers), dtype=dtype.float32)
        if as_tensor:
            registers = tensor.to_torch(registers, use_grad=True)
        return registers


class Repertoire(BaseRepertoire):
    def __init__(
        self,
        n_registers: int,
        n_functions: int,
        seed: int = 0,
        min_layers: int = 0,
        max_layers: int = 6,
        min_dims: int = 4,
        min_neurons: int = 8,
        max_neurons: int = 128,
        **kwargs
    ):
        params = OperatorParams(
            seed=seed,
            min_dims=min_dims,
            max_dims=n_registers,
            min_layers=min_layers,
            max_layers=max_layers,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
        )
        functions = nn.ModuleList(
            [Operator(seed=seed + i, **params.sample(**kwargs)) for i in range(n_functions)]
        )
        super(Repertoire, self).__init__(functions=functions, n_registers=n_registers)
