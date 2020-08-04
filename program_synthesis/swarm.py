from fragile.core import Swarm

from program_synthesis.walkers import ClassificationWalkers


class ClassificationSwarm(Swarm):
    """
    Swarm with default parameters for solving classification tasks with \
    a ProgramSynthesis environment.
    """

    def __init__(
        self,
        minimize: bool = True,
        accumulate_rewards: bool = False,
        distance_function=None,
        walkers=ClassificationWalkers,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`ClassificationSwarm`.

        Args:
            minimize: If ``True`` the algorithm will perform a minimization \
                      process. If ``False`` it will be a maximization process.
            accumulate_rewards: If ``True`` the rewards obtained after transitioning \
                                to a new state will accumulate. If ``False`` only the last \
                                reward will be taken into account.
            distance_function: Function to compute the distances between two \
                               groups of walkers. It will be applied row-wise \
                               to the walkers observations and it will return a \
                               vector of scalars. Defaults to l2 norm.
            walkers: A callable that returns an instance of :class:`BaseWalkers`.
            *args: Passed to :class:`Swarm` ``__init__``.
            **kwargs: Passed to :class:`Swarm` ``__init__``.

        """

        # The distance is calculated inside the environment for more effective parallelization.
        # This function only aggregates the results of the two dimensional observations
        # returned by the environment.
        def _distance(x, y):
            return x.mean(1)

        distance = _distance if distance_function is None else distance_function
        super(ClassificationSwarm, self).__init__(
            distance_function=distance,
            accumulate_rewards=accumulate_rewards,
            minimize=minimize,
            walkers=walkers,
            *args,
            **kwargs
        )
