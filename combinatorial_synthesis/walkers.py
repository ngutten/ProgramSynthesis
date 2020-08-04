from fragile.backend import tensor
from fragile.core.walkers import StatesWalkers, Walkers
import numpy


class ClassificationStatesWalkers(StatesWalkers):
    """StatesWalkers class that keeps track of additional performance metrics."""

    def __init__(self, *args, **kwargs):
        """Initialize a :class:`SynthesisStatesWalkers`."""
        super(ClassificationStatesWalkers, self).__init__(*args, **kwargs)
        self.best_train_acc = numpy.nan
        self.best_val_acc = numpy.nan
        self.best_train_loss = numpy.nan
        self.best_val_loss = numpy.nan


def update_best_classification(walkers: "ClassificationWalkers"):
    """Keep track of the best state found and its reward."""
    ix = walkers.get_best_index()
    best_obs = tensor.copy(walkers.env_states.observs[ix])
    best_reward = tensor.copy(walkers.states.cum_rewards[ix])
    best_state = tensor.copy(walkers.env_states.states[ix])
    best_is_in_bounds = not bool(walkers.env_states.oobs[ix])
    has_improved = (
        walkers.states.best_reward > best_reward
        if walkers.minimize
        else walkers.states.best_reward < best_reward
    )
    if has_improved and best_is_in_bounds:
        walkers.states.update(
            best_reward=best_reward,
            best_state=best_state,
            best_obs=best_obs,
            best_id=tensor.copy(walkers.states.id_walkers[ix]),
            best_train_acc=tensor.copy(walkers.env_states.train_acc[ix]),
            best_val_acc=tensor.copy(walkers.env_states.val_acc[ix]),
            best_train_loss=tensor.copy(walkers.env_states.train_loss[ix]),
            best_val_loss=tensor.copy(walkers.env_states.val_loss[ix]),
            best_time=tensor.copy(walkers.env_states.times[ix]),
        )


def fix_best_classification(walkers: "ClassificationWalkers"):
    """Ensure the best state found is assigned to the last walker of the \
    swarm, so walkers can always choose to clone to the best state."""
    if walkers.states.best_reward is not None and walkers.clone_to_best:
        walkers.env_states.observs[-1] = tensor.copy(walkers.states.best_obs)
        walkers.states.cum_rewards[-1] = tensor.copy(walkers.states.best_reward)
        walkers.states.id_walkers[-1] = tensor.copy(walkers.states.best_id)
        walkers.env_states.times[-1] = tensor.copy(walkers.states.best_time)
        walkers.env_states.states[-1] = tensor.copy(walkers.states.best_state)
        walkers.env_states.val_loss[-1] = tensor.copy(walkers.states.best_val_loss)
        walkers.env_states.train_loss[-1] = tensor.copy(walkers.states.best_train_loss)
        walkers.env_states.val_acc[-1] = tensor.copy(walkers.states.best_val_acc)
        walkers.env_states.train_acc[-1] = tensor.copy(walkers.states.best_train_acc)


class ClassificationWalkers(Walkers):
    """
    Walkers class that tracks the accuracy and loss of the best solution found \
    in the training and test set.
    """

    STATE_CLASS = ClassificationStatesWalkers

    def __repr__(self):
        return super(ClassificationWalkers, self).__repr__()

    def update_best(self):
        """Keep track of the best state found and its reward."""
        return update_best_classification(self)

    def fix_best(self):
        """Ensure the best state found is assigned to the last walker of the \
        swarm, so walkers can always choose to clone to the best state."""
        return fix_best_classification(self)
