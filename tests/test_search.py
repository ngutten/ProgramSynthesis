import pytest

from fragile.core.tree import HistoryTree
from fragile.distributed.env import RayEnv
import ray


from program_synthesis.env import ProgramSynthesis
from program_synthesis.datasets import make_wines, make_high_skewed_gaussian
from program_synthesis.model import ProgramSamplerNop
from program_synthesis.repertoire import Repertoire
from program_synthesis.swarm import ClassificationSwarm

from tests.test_swarm import TestSwarm

ray.init(ignore_reinit_error=True)

n_registers = 8
n_functions = 10
n_in_registers = 3
n_out_registers = 2
n_neurons_function = 5
n_intermediate_layers = 0


n_walkers = 64
n_neurons = 8
n_layers = 1

n_classes = 3
n_samples = 300
n_features = 8


def make_swarm(reward_limit, n_walkers, datase_func=make_high_skewed_gaussian, **kwargs):
    max_len = 10

    repertoire = Repertoire(
        n_registers=n_registers,
        min_dims=4,
        n_functions=n_functions,
        layers=n_layers,
        min_neurons=n_neurons,
        max_neurons=n_neurons * 2,
        seed=555,
    )
    env = ProgramSynthesis(
        repertoire=repertoire,
        output_dims=n_classes,
        dataset_func=datase_func,
        samples=n_samples,
        n_classes=n_classes,
        n_features=n_features,
        max_len=max_len,
        dataset_seed=160290,
    )

    env = RayEnv(env, n_workers=8)
    model = lambda env: ProgramSamplerNop(env=env)
    tree = HistoryTree(names=["actions"], prune=True)
    swarm = lambda: ClassificationSwarm(
        model=model,
        env=env,
        tree=tree,
        n_walkers=n_walkers,
        max_epochs=50,
        reward_limit=reward_limit,
        **kwargs
    )
    return swarm


def gaussian_synth():
    n_classes = 3
    n_features = 8
    samples = 240
    dataset_seed = 160290
    reward_limit = 0.26
    n_walkers = 100
    return make_swarm(
        n_walkers=n_walkers,
        reward_limit=reward_limit,
        samples=samples,
        n_classes=n_classes,
        n_features=n_features,
        dataset_seed=dataset_seed,
    )()


def wines_synth() -> ClassificationSwarm:
    return make_swarm(n_walkers=200, dataset_func=make_wines, reward_limit=0.25)()


swarm_dict = {
    "gaussian_3": gaussian_synth,
    "wines": wines_synth,
}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "gaussian_3": 0.26,
    "wines": 0.25,
}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    return swarm_dict.get(request.param)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    swarm = swarm_dict.get(request.param)()
    score = test_scores[request.param]
    return swarm, score
