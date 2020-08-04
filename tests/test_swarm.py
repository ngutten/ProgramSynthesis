import numpy
import pytest

swarm_dict = {}
swarm_names = [None]
test_scores = {}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    if request.param is None:
        return None
    return swarm_dict.get(request.param)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    if request.param is None:
        return None
    swarm = swarm_dict.get(request.param)()
    score = test_scores[request.param]
    return swarm, score


class TestSwarm:
    def test_repr(self, swarm):
        if swarm is None:
            return
        assert isinstance(swarm.__repr__(), str)

    def test_init_not_crashes(self, swarm):
        if swarm is None:
            return
        assert swarm is not None

    def test_env_init(self, swarm):
        if swarm is None:
            return
        assert hasattr(swarm.walkers.states, "will_clone")

    def test_reset_no_params(self, swarm):
        if swarm is None:
            return
        swarm.reset()

    def test_reset_with_root_walker(self, swarm):
        return
        swarm.reset()
        param_dict = swarm.walkers.env_states.get_params_dict()
        obs_dict = param_dict["observs"]
        state_dict = param_dict["states"]
        obs_size = obs_dict.get("size", obs_dict["shape"][1:])
        state_size = state_dict.get("size", state_dict["shape"][1:])
        obs = numpy.random.random(obs_size).astype(obs_dict["dtype"])
        state = numpy.random.random(state_size).astype(state_dict["dtype"])
        reward = 160290
        root_walker = OneWalker(observ=obs, reward=reward, state=state)
        swarm.reset(root_walker=root_walker)
        swarm_best_id = swarm.best_id
        root_walker_id = root_walker.id_walkers
        assert (swarm.best_obs == obs).all()
        assert (swarm.best_state == state).all()
        assert swarm.best_reward == reward
        assert swarm_best_id == root_walker_id
        assert (swarm.walkers.env_states.observs == obs).all()
        assert (swarm.walkers.env_states.states == state).all()
        assert (swarm.walkers.env_states.rewards == reward).all()
        assert (swarm.walkers.states.id_walkers == root_walker.id_walkers).all()

    def test_step_does_not_crashes(self, swarm):
        if swarm is None:
            return
        swarm.reset()
        swarm.step_walkers()

    def test_score_gets_higher(self, swarm_with_score):
        if swarm_with_score is None:
            return
        swarm, target_score = swarm_with_score
        swarm.walkers.seed(160290)
        swarm.reset()
        swarm.run()
        reward = (
            swarm.get("cum_rewards").min()
            if swarm.walkers.minimize
            else swarm.get("cum_rewards").max()
        )
        assert (
            reward <= target_score if swarm.walkers.minimize else reward >= target_score
        ), "Iters: {}, min_reward: {} rewards: {}".format(
            swarm.walkers.epoch,
            swarm.walkers.states.cum_rewards.min(),
            swarm.walkers.states.cum_rewards,
        )
