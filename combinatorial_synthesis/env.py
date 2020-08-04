from typing import Callable, Dict, Tuple

from fragile.backend import Backend, dtype, tensor, typing
from fragile.backend.functions.fractalai import relativize
from fragile.core import Environment
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import StateDict
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from combinatorial_synthesis.repertoire import BaseRepertoire
from combinatorial_synthesis.datasets import make_gaussian_datasets


class PSTransitioner:
    """
    The :class:`ProgramSynthesisTransitioner` class includes all the necessary \
    logic for managing the code synthesis functionality.

    This is a class that does not to be instantiated, and it works as a collection of repertoire
    that extend the :class:`ProgramSynthesis` environment. Separating all the logic used inside
    ``make_transitions`` this way allows for the class:`ProgramSynthesis` environment to be \
    stepped in parallel out of the box using :class:`fragile.ParallelEnv`.

    """

    @staticmethod
    def subsample_dataset(
        env, probs: typing.Tensor, train: bool = None,
    ):
        """Subsample the probabilities to match the training or test set if needed."""
        n_walkers = probs.shape[0]
        tgts = env.y if train is None else (env.y_train if train else env.y_test)
        # Account for the fact that we store the whole dataset inside each walker state
        # but we want to evaluate the probabilities of either the test set or the training set.
        inps = (
            probs
            if train is None
            else (probs[:, : env.train_split, :] if train else probs[:, env.train_split :, :])
        )
        n_examples = tgts.shape[0]
        tgts = tensor.tile(tensor.to_backend(tgts), n_walkers)
        inps = tensor.to_backend(inps)
        return inps, tgts, n_examples

    @staticmethod
    def get_init_registers(env: "MultiProgramClassification", n_walkers):
        registers = env.repertoire.get_empty_registers(
            n_walkers=n_walkers, batch_size=env.X.shape[0]
        )
        registers[:, :, : env.n_features] = env.X
        return registers

    @staticmethod
    def forward_programs(env, programs, registers, grad: bool = False):
        with Backend.use_backend("torch"):
            programs, registers = tensor.to_backend(programs), tensor.to_backend(registers)
            predictions = tensor.zeros((registers.shape[0], registers.shape[1], env.output_dims))
            for i, program in enumerate(programs):
                program_regs = tensor.unsqueeze(tensor.to_torch(registers[i]))
                outs = env.model.predict(
                    program=tensor.to_torch(program), registers=program_regs, grad=grad
                )
                predictions[i] = outs
        return tensor.to_backend(predictions)

    @staticmethod
    def update_registers(
        env,
        registers: typing.Tensor,
        actions: typing.Tensor,
        times: typing.Tensor,
        grad: bool = False,
        to_backend: bool = False,
    ) -> typing.Tensor:
        """
        Transition the states by updating the registers as described my the actions.

        Args:
            env: Environment used in the current program synthesis task.
            registers: Contain the writable registers for each walker.
            actions: Describes the register update process for each walker.
            grad: Compute the gradients of the model.

        Returns:
            Tensor containing the updated registers.

        """
        # Assumes actions are integers
        actions = actions.flatten()
        with Backend.use_backend("torch"):
            new_registers = tensor.zeros(registers.shape, dtype=dtype.float32)
            for walker_i, (action, time) in enumerate(zip(actions, times.copy())):
                if time >= env.max_len:
                    continue
                walker_regs = tensor.to_backend(registers[walker_i], use_grad=grad)
                # Calculate outputs and write them to the corresponding registers
                new_registers[walker_i, :] = env.repertoire.forward_one_action(
                    walker_regs, action=action, grad=grad, index=int(time)
                )
        if not grad and to_backend:
            new_registers = tensor.to_backend(new_registers)
        return new_registers

    @staticmethod
    def update_actions(states: typing.Tensor, actions: typing.Tensor, target_prog):
        """Add the last sampled actions to the current state representing the action history."""

        step_ixs = tensor.argmax(
            states[:, :, target_prog] < 0, 1
        )  # Index of current step in program
        terminals = tensor.logical_not((states[:, :, target_prog] < 0).any(1))
        for i, idx in enumerate(step_ixs):
            if not terminals[i]:
                states[i, idx, target_prog] = actions[i]
        return states, terminals

    @staticmethod
    def process_obs(env, probs: typing.Tensor, states: typing.Tensor,) -> typing.Tensor:
        """
        Calculate the distances of the registers and the probability distributions \
        provided by the model.
        """

        def euclidean_distance(x, y):
            return tensor.sqrt(tensor.sum((x - y) ** 2, axis=1))

        n_walkers = states.shape[0]
        compas = env.random_state.permutation(tensor.arange(n_walkers))
        actions_hist = states.reshape(n_walkers, -1)
        probs = probs.reshape(n_walkers, -1)
        dist_probs = relativize(euclidean_distance(probs, probs[compas]))
        dist_states = relativize(euclidean_distance(actions_hist, actions_hist[compas]))

        observs = tensor.empty((n_walkers, 3), dtype=dtype.float32)
        observs[:, 0] = dist_states
        observs[:, 1] = dist_probs
        observs[:, 2] = 1.0
        return observs

    @staticmethod
    def boundary_condition(env, rewards: typing.Tensor, times: typing.Tensor) -> typing.Tensor:
        """
        Apply an arbitrary boundary conditions to discard ill-performing states.

        It discards states with losses greater than the mean loss of all walkers \
        plus one standard deviation.
        """
        mean, std = rewards.mean(), rewards.std()
        too_bad = rewards > (mean + std)
        too_long = times > env.max_len - 1  # if gym_env.max_len is not None else too_bad
        oobs = tensor.logical_or(too_bad, too_long)
        if not oobs.all():
            return oobs
        return tensor.zeros(rewards.shape[0], dtype=dtype.bool)

    @staticmethod
    def calculate_output(env, registers: typing.Tensor) -> typing.Tensor:
        """Run the output fully connected layer using the registers as input."""
        n_walkers = registers.shape[0]
        batch_size = int(registers.shape[1])
        probs = env.model.calculate_output(registers)
        probs = probs.reshape(n_walkers, batch_size, env.output_dims)
        return probs

    @classmethod
    def calculate_loss(
        cls, env, probs: typing.Tensor, train: bool = None, grad: bool = False,
    ) -> typing.Tensor:
        """
        Calculate the cross entropy loss of the output layer with respect \
        to the current dataset.
        """
        inps, tgts, n_examples = PSTransitioner.subsample_dataset(env, probs=probs, train=train)

        n_walkers = probs.shape[0]
        targets = tensor.to_torch(tgts)
        probs_tensor = tensor.to_torch(inps.reshape(-1, env.n_classes))
        losses = env.loss_func(probs_tensor, targets)
        losses = losses.reshape((n_walkers, n_examples))
        losses = losses.mean(1).flatten()
        if not grad:
            losses = tensor.to_backend(losses)
        return losses

    @classmethod
    def calculate_accuracy(cls, env, probs: typing.Tensor, train: bool = None,) -> typing.Tensor:
        """Calculate the accuracy of the output layer for predicting the dataset classes."""
        inps, tgts, n_examples = cls.subsample_dataset(env, probs=probs, train=train,)
        n_walkers = probs.shape[0]
        inps, tgts = tensor.to_numpy(inps), tensor.to_numpy(tgts)
        inps = inps.reshape(-1, env.n_classes)
        acc = inps.argmax(1) == tgts
        acc = acc.reshape(n_walkers, n_examples).sum(1) / n_examples
        return acc


class MultiprogramClassificationModel(nn.Module):
    def __init__(self, output_dim, repertoire: BaseRepertoire, freeze_mappings: bool = False):
        super(MultiprogramClassificationModel, self).__init__()
        self.output_dim = output_dim
        self.repertoire = repertoire
        self.loss_function = nn.CrossEntropyLoss(reduction="none")
        self.target_class = 0
        self.one_dim = False
        self.out_heads = torch.nn.ModuleList(
            [torch.nn.Linear(self.repertoire.n_registers, 1) for _ in range(output_dim)]
        )
        if freeze_mappings:
            for head in self.out_heads:
                head.requires_grad_(False)

    def calculate_output(self, registers, index=0):
        output = self.out_heads[index](registers)
        return output

    def predict_one(self, registers, program, grad: bool = False, index=0):
        _registers = self.repertoire.forward(registers=registers, program=program, grad=grad)
        probs = self.calculate_output(_registers, index=index)
        return probs

    def predict(self, registers, program, grad: bool = False):
        if self.one_dim:
            return self.predict_one(
                registers, program[:, self.target_class], grad=grad, index=self.target_class
            )
        predictions = [
            self.predict_one(registers, program[:, i], grad=grad, index=i)
            for i in range(self.output_dim)
        ]
        return tensor.concatenate(predictions, 0).T


class ProgramSynthesis(Environment):
    """
    Environment that represents a program synthesis task for gaussian distributions classification.

    It defines a dataset, a predefined number of registers, and a list of repertoire \
    (Randomly initialized neural networks). Each action represents reading two registers \
    (either a feature of the dataset, or a previously written register),
    applying a function to them, and saving the output in a new register.

    The classification probabilities are provided by a randomly initialized fully \
    connected layer that reads the registers where the environment can write.

    """

    def __init__(
        self,
        repertoire: BaseRepertoire,
        output_dims: int,
        n_features: int,
        max_len: int = 10000,
        observs_shape=None,
        batch_size=None,
        dataset_seed: float = None,
        train_split: float = 0.75,
        freeze_mappings: bool = False,
        dataset_func: Callable[..., Tuple[typing.Tensor, typing.Tensor]] = make_gaussian_datasets,
        *args,
        **kwargs,
    ):
        """
        Initialize a :class:`ProgramSynthesis`.

        Args:
            output_dims: Number of dimensions of the output layer.
            n_features: Number of read only registers. Equals to the number of \
                         features of the target task's data.
            max_len: Maximum number of actions per program.


        """
        self.n_actions = None
        self.max_len = max_len
        self.repertoire = repertoire
        self.repertoire.n_output = output_dims
        self.n_registers = self.repertoire.n_registers
        self.n_features = n_features
        self.output_dims = output_dims
        self.dataset_seed = dataset_seed
        self.train_split = train_split
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test = [None] * 6
        self.dataset_func = dataset_func
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        self.make_datasets()
        self.n_examples = len(self.y)
        self.batch_size = self.n_examples
        self.n_classes = len(tensor.unique(self.y.flatten()))
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        observs_shape = (3,) if observs_shape is None else observs_shape
        states_shape = (max_len, self.n_classes)
        super(ProgramSynthesis, self).__init__(
            states_shape=states_shape, observs_shape=observs_shape
        )
        self.batch_size = batch_size if batch_size is not None else self.n_examples
        self.model = MultiprogramClassificationModel(
            output_dim=self.n_classes, repertoire=repertoire, freeze_mappings=freeze_mappings,
        )

    def import_dataset(self, X, y, split):
        self.X = X
        self.y = y
        self.X_train = X[:split]
        self.X_test = X[split:]
        self.y_train = y[:split]
        self.y_test = y[split:]
        self.train_split = split

    def make_datasets(self):
        """Create the dataset that will be used for training the model."""
        X, y = self.dataset_func(seed=self.dataset_seed, *self.dataset_args, **self.dataset_kwargs)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=self.train_split, random_state=self.dataset_seed
        )
        self.X = tensor.concatenate([self.X_train, self.X_test], axis=0)
        self.y = tensor.concatenate([self.y_train, self.y_test])
        self.train_split = len(self.y_train)

    def calculate_loss(
        self, probs: typing.Tensor, train: bool = None, grad: bool = False,
    ) -> typing.Tensor:
        """
        Calculate the cross entropy loss of the output layer with respect \
        to the current dataset.
        """
        return PSTransitioner.calculate_loss(self, probs=probs, train=train, grad=grad)

    def make_transitions(
        self,
        states: typing.Tensor,
        actions: typing.Tensor,
        selected_hist: typing.Tensor = None,
        times: typing.Tensor = None,
    ) -> Dict[str, typing.Tensor]:
        """
        Apply the actions to the provided states, and calculate the \
        metrics and boundary conditions of the new states.
        """
        with torch.no_grad():
            new_states, terminals = PSTransitioner.update_actions(
                states=tensor.copy(states),
                actions=actions,
                target_prog=np.random.randint(0, self.n_classes),
            )
            registers = PSTransitioner.get_init_registers(self, states.shape[0])
            probs = PSTransitioner.forward_programs(self, new_states, registers)
            train_loss = PSTransitioner.calculate_loss(self, probs, train=True, grad=False)
            train_acc = PSTransitioner.calculate_accuracy(self, probs, train=True,)
            val_loss = PSTransitioner.calculate_loss(self, probs=probs, train=False)
            val_acc = PSTransitioner.calculate_accuracy(self, probs=probs, train=False)
            observs = PSTransitioner.process_obs(self, probs, new_states)
            oobs = PSTransitioner.boundary_condition(self, train_loss, times)
            data = {
                "states": tensor.to_backend(new_states),
                "observs": tensor(observs),
                "rewards": tensor.to_backend(train_loss),
                "train_loss": tensor.to_backend(train_loss),
                "val_loss": tensor.to_backend(val_loss),
                "train_acc": tensor.to_backend(train_acc),
                "val_acc": tensor.to_backend(val_acc),
                "oobs": tensor.to_backend(oobs),
                "terminals": tensor.to_backend(terminals),
            }
            return data

    def get_params_dict(self) -> StateDict:
        """
        Return a dictionary containing the param_dict to build an instance \
        of :class:`StatesEnv` that can handle all the data generated by an \
        :class:`ProgramSynthesis`.
        """
        params = {
            "states": {"size": self.states_shape, "dtype": dtype.int32},
            "observs": {"size": self.observs_shape, "dtype": dtype.float32},
            "rewards": {"dtype": dtype.float32},
            "times": {"dtype": dtype.int32},
            "oobs": {"dtype": dtype.bool},
            "terminals": {"dtype": dtype.bool},
            "val_loss": {"dtype": dtype.float32},
            "train_loss": {"dtype": dtype.float32},
            "train_acc": {"dtype": dtype.float32},
            "val_acc": {"dtype": dtype.float32},
        }
        return params

    def states_to_data(
        self, model_states: StatesModel, env_states: StatesEnv
    ) -> Dict[str, typing.Tensor]:
        """
        Extract the data that will be used to make the state transitions.

        Args:
            model_states: :class:`StatesModel` representing the data to be used \
                         to act on the environment.
            env_states: :class:`StatesEnv` representing the data to be set in \
                       the environment.

        Returns:
            Dictionary containing:

            ``{"states": np.array, "actions": np.array}``

        """
        data = {
            "states": tensor(env_states.states),
            "actions": tensor(model_states.actions),
            "times": tensor(env_states.times),
        }
        return data

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the environment to the start of a new episode and return a new \
        :class:`StatesEnv` instance describing the state of the :envs:`Environment`.

        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`StatesEnv` instance describing the state of the Environment. The first \
            dimension of the data tensors (number of walkers) will be equal to \
            batch_size.

        """
        states = tensor.ones(tuple([batch_size]) + self.states_shape, dtype=dtype.int32) * -1
        observs = tensor.zeros((batch_size, 3))
        rewards = tensor.ones(batch_size, dtype=dtype.float32) * numpy.inf
        oobs = tensor.zeros(batch_size, dtype=dtype.bool)
        times = tensor.zeros(batch_size, dtype=dtype.int32)
        new_states = self.states_from_data(
            batch_size=batch_size,
            states=states,
            observs=observs,
            rewards=rewards,
            oobs=oobs,
            times=times,
        )
        return new_states
