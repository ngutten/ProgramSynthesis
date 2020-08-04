import warnings

from fragile.distributed import RayEnv
import numpy as np
import pandas as pd
import ray
import torch
from tqdm.autonotebook import trange


from combinatorial_synthesis.env import ProgramSynthesis
from combinatorial_synthesis.datasets import make_sinusoids, make_rastrigin
from combinatorial_synthesis.model import ProgramSamplerNop
from combinatorial_synthesis.repertoire import Repertoire
from combinatorial_synthesis.swarm import ClassificationSwarm
from combinatorial_synthesis.training import exchange_operators, train_agents, ModuleTrainer

ray.init(ignore_reinit_error=True)

warnings.filterwarnings("ignore")


def create_swarm(n_walkers, dataset_function, random_operators: bool = False, seed=0):
    n_registers = 12
    n_functions = 15
    n_neurons = 32
    n_layers = 2
    n_classes = 3
    n_samples = 10000
    n_features = 8
    max_len = 10

    repertoire_loaded = torch.load("transfer_learning_bo_gaussian_train")
    if random_operators:
        for f in repertoire_loaded.functions:
            for l in f.layers:
                torch.nn.init.orthogonal_(l.weight)
    repertoire = Repertoire(
        n_registers=n_registers,
        min_dims=3,
        n_functions=n_functions,
        layers=n_layers,
        min_neurons=n_neurons,
        max_neurons=n_neurons * 2,
        seed=555 + seed * 1000,
    )
    repertoire.functions = repertoire_loaded.functions
    env = ProgramSynthesis(
        repertoire=repertoire,
        dataset_func=dataset_function,
        samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        max_len=max_len,
        dataset_seed=160290 + seed * 1000,
    )
    env = RayEnv(env, 64)

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


def find_and_train_one_program(
    dataset_func, random_operators: bool = True, freeze_mappings: bool = False, run_name=""
):
    pop_size = 1
    n_walkers = 64
    metabatch = 1
    n_epochs = 1
    population = [
        create_swarm(
            n_walkers=n_walkers,
            seed=i,
            dataset_function=dataset_func,
            random_operators=random_operators,
        )
        for i in range(pop_size)
    ]
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
                task_id = evaluate_swarm(
                    swarm=s,
                    agent_index=i,
                    prog_ix=j,
                    epoch_ix=epoch,
                    seed=np.random.randint(1000000),
                )
                task_ids.append(task_id)
        results = task_ids
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

        all_train_losses = []
        for _ in trange(1200):
            train_losses = train_agents(
                trainers, agent_ix, programs, train_data, target_data, dataset_splits, program_ids
            )
            all_train_losses.extend(train_losses)
        pd.DataFrame(all_losses).to_csv("training_single_{}.csv".format(run_name))
        return all_train_losses


if __name__ == "__main__":

    find_and_train_one_program(
        dataset_func=make_rastrigin,
        random_operators=True,
        freeze_mappings=True,
        run_name="random_rastrigin_freeze",
    )
    find_and_train_one_program(
        dataset_func=make_rastrigin,
        random_operators=True,
        freeze_mappings=False,
        run_name="random_rastrigin_nofreeze",
    )
    find_and_train_one_program(
        dataset_func=make_rastrigin,
        random_operators=False,
        freeze_mappings=True,
        run_name="pretrained_rastrigin_freeze",
    )
    find_and_train_one_program(
        dataset_func=make_rastrigin,
        random_operators=False,
        freeze_mappings=False,
        run_name="pretrained_rastrigin_nofreeze",
    )
