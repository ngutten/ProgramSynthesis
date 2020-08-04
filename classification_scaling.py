import warnings

import pandas as pd
import numpy as np
from tqdm.autonotebook import trange
import ray

from combinatorial_synthesis.env import ProgramSynthesis
from combinatorial_synthesis.datasets import make_high_skewed_gaussian
from combinatorial_synthesis.model import ProgramSamplerNop
from combinatorial_synthesis.repertoire import Repertoire
from combinatorial_synthesis.swarm import ClassificationSwarm
from combinatorial_synthesis.training import exchange_operators, train_agents, ModuleTrainer

ray.init(ignore_reinit_error=True)

warnings.filterwarnings("ignore")


def create_swarm(n_walkers, seed=0):
    n_registers = 12
    n_functions = 15
    n_neurons = 32
    n_layers = 2
    n_classes = 6
    n_samples = 500
    n_features = 8
    max_len = 10

    repertoire = Repertoire(
        n_registers=n_registers,
        min_dims=4,
        n_functions=n_functions,
        layers=n_layers,
        min_neurons=n_neurons,
        max_neurons=n_neurons * 2,
        seed=555 + seed * 1000,
    )
    env = ProgramSynthesis(
        repertoire=repertoire,
        dataset_func=make_high_skewed_gaussian,
        samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        output_dims=n_classes,
        max_len=max_len,
        dataset_seed=160290 + seed * 1000,
    )

    model = ProgramSamplerNop(env)
    swarm = ClassificationSwarm(
        env=env,
        model=model,
        n_walkers=n_walkers,
        max_epochs=100,
        fix_best=True,
        use_notebook_widget=False,
        show_pbar=False,
    )
    return swarm


@ray.remote
def evaluate_swarm(swarm, agent_index, prog_ix, epoch_ix, seed):
    swarm.env.dataset_seed = seed
    swarm.env.make_datasets()
    swarm.run()
    metrics = {
        "index": "%s_%s_%s" % (epoch_ix, agent_index, prog_ix),
        "agent": int(agent_index),
        "run": int(prog_ix),
        "epoch": int(epoch_ix),
        "seed": int(seed),
        "train_loss": float(swarm.best_reward),
        "train_acc": float(swarm.walkers.states.best_train_acc),
        "val_loss": float(swarm.walkers.states.best_val_loss),
        "val_acc": float(swarm.walkers.states.best_val_acc),
    }
    return (
        metrics,
        swarm.best_state.copy(),  # best program found
        swarm.env.X.copy(),
        swarm.env.y.copy(),
        int(swarm.env.train_split),
    )


def scaling_experiment():
    pop_size = 1
    n_walkers = 64
    metabatch = 32
    n_epochs = 150
    population = [create_swarm(n_walkers=n_walkers, seed=i) for i in range(pop_size)]
    trainers = [ModuleTrainer(swarm) for swarm in population]

    all_metrics = []
    all_losses = []
    all_programs = []

    for epoch in trange(n_epochs):
        metrics = []
        programs = []
        train_data = []
        target_data = []
        dataset_splits = []
        task_ids = []
        swarm_ix = []
        program_ids = []
        # Sample programs and keep track o metrics and datasets
        # Create ray tasks
        for i, s in enumerate(population):
            for j in range(metabatch):
                task_id = evaluate_swarm.remote(
                    swarm=s,
                    agent_index=i,
                    prog_ix=j,
                    epoch_ix=epoch,
                    seed=np.random.randint(1000000),
                )
                task_ids.append(task_id)
        results = ray.get(task_ids)
        # Retreive results and data wrangling
        for mets, prog, X, y, split in results:
            swarm_ix.append(int(mets["agent"]))
            program_ids.append(str(mets["index"]))
            metrics.append(mets)
            programs.append(prog.copy())
            train_data.append(X.copy())
            target_data.append(y.copy())
            dataset_splits.append(int(split))
        all_metrics.extend(metrics)
        agent_ix = np.array(swarm_ix)
        programs = np.array(programs)
        train_data = np.array(train_data)
        target_data = np.array(target_data)
        dataset_splits = np.array(dataset_splits)
        program_ids = np.array(program_ids)
        all_programs.append(programs)
        losses = train_agents(
            trainers, agent_ix, programs, train_data, target_data, dataset_splits, program_ids
        )
        all_losses.extend(losses)
        exchange_operators(population, programs, agent_ix)
        df = pd.DataFrame(all_metrics)
        df.to_csv("scaling_{}w_{}pop.csv".format(n_walkers, pop_size))


if __name__ == "__main__":
    scaling_experiment()
