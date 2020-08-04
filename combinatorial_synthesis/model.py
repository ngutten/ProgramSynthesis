from fragile.backend import dtype, random_state, tensor, typing
from fragile.core.models import DiscreteModel
from fragile.core.states import StatesModel


from combinatorial_synthesis.env import ProgramSynthesis


class ProgramSampler(DiscreteModel):
    """
    Samples the actions that define a program synthesis problem.

    Each action consists on 4 discrete values: ``[r_1, r_2, w, func]``.

    - r_1: Describes the first register that will be used as an input for the function.
           It can be either a dataset feature of a writable register.
    - r_2: Describes the second register that will be used as an input for the function.
           It can be either a dataset feature of a writable register.
    - w: Describes the register where the output of the function will be written.
           It is the index of the target writable register.
    - func: The index of the available function that will be applied to r_1 and r_2.

    """

    def __init__(self, env: ProgramSynthesis):
        """
        Initialize a :class:`ProgramSampler`.

        Args:
            env: Environment describing the target program synthesis task.

        """
        super(ProgramSampler, self).__init__(critic=None, env=env, n_actions=None)
        self.n_functions = env.repertoire.n_functions
        self.max_len = len(env.repertoire)

    def get_params_dict(self, override_params: bool = True) -> typing.StateDict:
        """Return a dictionary describing the data sampled by the model."""
        params = super(ProgramSampler, self).get_params_dict()
        pdict = {"actions": {"dtype": dtype.int32}}
        params.update(pdict)
        return params

    def sample(self, batch_size: int, model_states: StatesModel = None, **kwargs) -> StatesModel:
        """
        Sample an array of 4 discrete variables from a uniform prior.

        The values of each dimension of the action array will correspond to the available
        actions for reading register, writing a register, of applying a function.

        See the class :class:`ProgramSampler` docstring for more information.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            kwargs: passed to the :class:`Critic`.

        Returns:
            :class:`States` variable containing the calculated actions and dt.

        """
        # Sample actions for read write and function from discrete uniform distribution

        actions = random_state.randint(0, self.n_functions, size=batch_size)
        actions = tensor.astype(actions, dtype=dtype.int32)
        return self.update_states_with_critic(
            actions=actions, model_states=model_states, batch_size=batch_size, **kwargs
        )


class ProgramSamplerNop(ProgramSampler):
    def __init__(self, env, nop_prob: float = 0.1):
        super(ProgramSamplerNop, self).__init__(env)
        self.nop_prob = nop_prob

    def sample(self, batch_size: int, model_states: StatesModel = None, **kwargs) -> StatesModel:
        """
        Sample an array of 4 discrete variables from a uniform prior.

        The values of each dimension of the action array will correspond to the available
        actions for reading register, writing a register, of applying a function.

        See the class :class:`ProgramSampler` docstring for more information.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            kwargs: passed to the :class:`Critic`.

        Returns:
            :class:`States` variable containing the calculated actions and dt.

        """
        # Sample actions for read write and function from discrete uniform distribution

        actions = random_state.randint(0, self.n_functions, size=batch_size)
        actions = tensor.where(random_state.random(actions.shape) < self.nop_prob, -1.0, actions)
        actions = tensor.astype(actions, dtype=dtype.int32)
        return self.update_states_with_critic(
            actions=actions, model_states=model_states, batch_size=batch_size, **kwargs
        )
