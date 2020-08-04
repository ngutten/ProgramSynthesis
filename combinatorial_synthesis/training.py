from fragile.backend import Backend, tensor
from fragile.core import Swarm
import numpy as np
import torch

from combinatorial_synthesis.env import ProgramSynthesis, PSTransitioner
from combinatorial_synthesis.modules import Operator


class AdamOptimizer(torch.nn.Module):
    def __init__(self, model, lr=0.001, weight_decay=0.0):
        super(AdamOptimizer, self).__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class ModuleTrainer:
    def __init__(self, swarm: Swarm, optimizer=AdamOptimizer, **kwargs):
        self.env: ProgramSynthesis = swarm.env
        self.minimize = swarm.walkers.minimize
        self.optimizer = optimizer(self.env.model, **kwargs)
        self.epochs = 0

    def get_loss(self, program, X, y, split):
        self.env.import_dataset(X, y, split)
        registers = PSTransitioner.get_init_registers(self.env, 1)
        with Backend.use_backend("torch"):
            registers = tensor.to_backend(registers)
            preds = self.env.model.predict(program=program, registers=registers, grad=True)
            loss = self.env.calculate_loss(preds, train=True, grad=True)
        return loss

    def train(self, programs, dataset_Xs, dataset_ys, dataset_splits, indexes):
        losses = []
        self.optimizer.zero_grad()
        for prog, X, y, split, index in zip(
            programs, dataset_Xs, dataset_ys, dataset_splits, indexes
        ):
            data_loss = self.get_loss(prog, X, y, split)
            loss = data_loss.mean()
            loss.backward()
            losses.append({"index": index, "loss": loss.cpu().detach().item()})
        for p in self.env.repertoire.parameters():
            if p.grad is not None:
                p.grad.mul_(1.0 / len(programs))
        self.optimizer.step()
        self.epochs += 1
        return losses


def train_agents(trainers, agent_ix, programs, train_data, target_data, dataset_splits, indexes):
    losses = []
    for i in set(agent_ix):
        ix = agent_ix == i  # train a batch of programs from the same agent
        l = trainers[i].train(
            programs[ix], train_data[ix], target_data[ix], dataset_splits[ix], indexes[ix]
        )
        losses.extend(l)
    return losses


def count_operations(programs, n_ops):
    unique, counts = np.unique(programs.flatten(), return_counts=True)
    freqs = np.zeros(n_ops)
    freqs[unique[unique >= 0]] = counts[unique >= 0]
    return freqs


def rankfit(x):
    npx = np.array(x)
    ranks = np.zeros(npx.shape[0])
    ranks[np.argsort(npx)] = np.arange(npx.shape[0])
    ranks = 2 * (ranks / npx.shape[0] - 0.5)
    return ranks


def get_op_form_compa(programs, agent_ix, ix, n_ops):
    """Select the operation that will be imported from another agent."""
    op_stats = count_operations(programs[agent_ix == ix], n_ops)
    proba = np.exp(4 * rankfit(op_stats))
    proba = proba / np.sum(proba)
    k = np.random.choice(np.arange(proba.shape[0]), p=proba)
    return k


def exchange_operators(population, programs, agent_ix):
    for i, swarm in enumerate(population):
        n_ops = len(swarm.env.repertoire)
        freqs = count_operations(programs[agent_ix == i], n_ops)
        exchange_ops = np.arange(n_ops)[freqs < 0.75 * freqs.sum() / n_ops]
        other_swarms = np.arange(len(population))
        # It is possible to exchange operators with itself only when population size = 1
        other_swarms = other_swarms[other_swarms != i] if len(population) > 1 else [0]
        for op in exchange_ops:
            other_ix = np.random.choice(other_swarms)
            n_ops = len(population[other_ix].env.repertoire)
            new_op_ix = get_op_form_compa(programs, agent_ix, other_ix, n_ops)
            target_op = population[other_ix].env.repertoire.functions[new_op_ix]
            exchanged_operator = Operator(**target_op.params)
            exchanged_operator.load_state_dict(target_op.state_dict())
            swarm.env.repertoire.functions[op] = exchanged_operator
