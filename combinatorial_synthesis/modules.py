from math import sqrt

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """
    Basic building block used for program synthesis.

    It is a function consisting on a randomly initialize fully connected layers \
    applied to two different registers. The output can be used as a new register value.

    Each layer besides the output layer is followed by a leaky_relu activation function.

    """

    def __init__(
        self,
        n_registers: int,
        seed=None,
        n_neurons: int = 8,
        n_layers: int = 2,
        output_func=torch.tanh,
        skip_connections: bool = True,
    ):
        """
        Initialize a :class:`Mapper`.

        Args:
            seed: Random seed for initializing the fully connected layers.
            n_registers: Number of registers registers.
            n_neurons: Number of neurons of each layer.
            n_layers: Number of internal fully connected layers besides the \
                      input and the output layer. For example, a value of 2 will \
                      create a Module with 4 layers: Input, fc1, fc2 and output.
            output_func: Activation function that will be applied to the output layer.

        """
        super().__init__()
        self.skip_connections = skip_connections
        self.seed = seed
        self.output_func = output_func
        self.n_dims = n_registers
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        if self.seed is not None:
            torch.manual_seed(seed)
        self.input = nn.Linear(self.n_dims, n_neurons)
        nn.init.orthogonal_(self.input.weight, gain=sqrt(2))
        self._names = []
        # Create n_layers fully connected layers named li to be executed
        # between input and output layers
        for i in range(n_layers):
            name = "l%s" % i
            layer = nn.Linear(n_neurons, n_neurons)
            nn.init.orthogonal_(layer.weight, gain=sqrt(2))
            setattr(self, name, layer)
            self._names.append(name)
        self.output = nn.Linear(n_neurons, self.n_dims)
        nn.init.orthogonal_(self.output.weight, gain=sqrt(2))

    def forward(self, x, *args, grad: bool = True):
        """Apply the model to the input registers."""
        if grad:
            return self._forward(x, *args)
        else:
            with torch.no_grad():
                return self._forward(x, *args)

    def _forward(self, x, *args):
        data = [x] + list(args) if args else [x]
        z = torch.cat(data, 1)
        z_in = z
        z = torch.tanh(self.input(z))
        for name in self._names:
            layer = getattr(self, name)
            z = torch.tanh(layer(z))
        out = self.output(z)
        if self.skip_connections:
            out = out + z_in
        if self.output_func is not None:
            out = self.output_func(out)
        return out


class NonLinearities:
    names = ("leaky_relu", "tanh", "elu", "sin")

    def __init__(self):
        self.leaky_relu = F.leaky_relu
        self.tanh = torch.tanh
        self.elu = F.elu
        self.sin = torch.sin

    def __getitem__(self, item):
        if not isinstance(item, str):
            item = self.names[item]
        return getattr(self, item)


class OperatorParams:
    def __init__(
        self,
        max_dims,
        seed: int = 0,
        min_layers: int = 0,
        max_layers: int = 6,
        min_dims: int = 4,
        min_neurons: int = 8,
        max_neurons: int = 128,
    ):

        self.random_state = numpy.random.RandomState(seed=seed)
        self.max_dims = max_dims
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.min_dims = min(min_dims, max_dims)
        self.nonlinearities = NonLinearities.names

    def sample(
        self, n_dims=None, n_neurons=None, layers=None, skip=None, sparse=None, nonlinearity=None,
    ):
        if n_dims is None:
            n_dims = self.random_state.randint(self.min_dims, self.max_dims)
        if n_neurons is None:
            n_neurons = self.random_state.randint(self.min_neurons, self.max_neurons)
        if layers is None:
            layers = self.random_state.randint(self.min_layers, self.max_layers)
        if skip is None:
            skip = self.random_state.random() > 0.5
        if sparse is None:
            sparse = self.random_state.random() > 0.5
        if nonlinearity is None:
            nonlinearity = self.random_state.choice(self.nonlinearities)

        params = {
            "n_dims": n_dims,
            "n_neurons": n_neurons,
            "layers": layers,
            "skip": skip,
            "sparse": sparse,
            "nonlinearity": nonlinearity,
        }
        return params


class Operator(nn.Module):
    def __init__(self, n_dims, n_neurons, layers, skip, sparse, nonlinearity, seed: int = None):
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(seed)
        self.n_dims = n_dims
        self.n_neurons = n_neurons
        self.n_layers = layers
        self.skip = skip
        self.sparse = sparse
        self.nonlinearity = NonLinearities()[nonlinearity]
        self._nonlin_name = nonlinearity
        super(Operator, self).__init__()
        if layers > 1:
            self.layers = nn.ModuleList(
                [nn.Linear(self.n_dims, self.n_neurons)]
                + [nn.Linear(self.n_neurons, self.n_neurons) for _ in range(layers - 2)]
                + [nn.Linear(self.n_neurons, self.n_dims)]
            )
        else:
            self.layers = nn.ModuleList([nn.Linear(self.n_dims, self.n_dims)])

        for l in self.layers:
            nn.init.orthogonal_(l.weight, gain=sqrt(2))

    @property
    def params(self):
        return {
            "n_dims": self.n_dims,
            "n_neurons": self.n_neurons,
            "layers": self.n_layers,
            "skip": self.skip,
            "sparse": self.sparse,
            "nonlinearity": self._nonlin_name,
        }

    def forward(self, x):
        z = x[:, : self.n_dims]
        for i in range(self.n_layers):
            if i == len(self.layers) - 1:
                z = self.layers[i](z)
            else:
                z = self.nonlinearity(self.layers[i](z))
        if self.skip:
            z = x[:, : self.n_dims] + z
        res = torch.cat([z, x[:, self.n_dims :]], 1)
        # normed = (res - res.mean(-1).unsqueeze(-1)) / res.std(-1).unsqueeze(-1)
        return res  # normed / torch.norm(normed, -1).unsqueeze(-1)


class A2COperator(Operator):
    def __init__(self, action_dims, obs_dims, *args, **kwargs):
        super(A2COperator, self).__init__(*args, **kwargs)
        self.action_dims = action_dims
        self.n_dims = action_dims + obs_dims
        self.actor = nn.Linear(self.n_dims, self.action_dims)
        self.critic = nn.Linear(self.n_dims, 1)
        self.model = nn.Linear(self.n_dims, obs_dims)

    def act(self, x):
        z = x[:, : self.n_dims]
        for i in range(self.n_layers):
            if i == len(self.layers) - 1:
                z = self.layers[i](z)
            else:
                z = self.nonlinearity(self.layers[i](z))
        if self.skip:
            z = x[:, : self.n_dims] + z
        z = self.actor(z)
        return z

    def evaluate(self, x):
        z = x[:, : self.n_dims]
        for i in range(self.n_layers):
            if i == len(self.layers) - 1:
                z = self.layers[i](z)
            else:
                z = self.nonlinearity(self.layers[i](z))
        if self.skip:
            z = x[:, : self.n_dims] + z
        z = self.critic(z)
        return z

    def apply_model(self, x):
        z = x[:, : self.n_dims]
        for i in range(self.n_layers):
            if i == len(self.layers) - 1:
                z = self.layers[i](z)
            else:
                z = self.nonlinearity(self.layers[i](z))
        if self.skip:
            z = x[:, : self.n_dims] + z
        z = self.model(z)
        return z
