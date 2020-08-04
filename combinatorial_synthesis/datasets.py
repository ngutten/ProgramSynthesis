import os
import pickle

from fragile.backend import tensor
from fragile.optimize.benchmarks import rastrigin
import numpy
from sklearn.datasets import load_digits, load_wine, make_swiss_roll
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer


def make_gaussian_datasets(
    samples: int, n_classes: int = 3, n_features: int = 8, one_hot_target: bool = False, seed=None,
):
    """
    Create gaussian distributions to be used as a toy dataset for classification problems.

    Args:
        samples: Number of total examples of the dataset.
        n_classes: Number of different classes of the dataset. Each class is \
                   sampled from a gaussian distribution with different means.
        n_features: Number of features of the dataset.
        one_hot_target: Return the class labels as one-hot encoded numpy arrays.
        seed: Random seed for generating the dataset.

    Returns:
            X, y tuple containing the features and the corresponding label for each example.

    """

    random_state = numpy.random.RandomState(seed)
    y = numpy.mod(numpy.arange(samples), n_classes)  # Balanced number of examples per class
    mu = random_state.randn(n_classes, n_features)  # Sample a mean vector for every class
    x = mu[y] + 0.5 * random_state.randn(samples, n_features)  # Add noise to all the samples

    y = (
        OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1)).astype(numpy.float32)
        if one_hot_target
        else y.astype(numpy.int64)
    )
    x = tensor.to_backend(x.astype(numpy.float32))
    return x, tensor.to_backend(y)


def make_high_skewed_gaussian(
    samples: int = 300,
    n_classes: int = 3,
    n_features: int = 8,
    one_hot_target: bool = False,
    seed=None,
):
    random_state = numpy.random.RandomState(seed)
    y = numpy.mod(numpy.arange(samples), n_classes)
    mu = random_state.randn(n_classes, n_features)
    skews = 0.05 * numpy.eye(n_features, n_features)[
        numpy.newaxis, :, :
    ] + 0.25 * random_state.randn(n_classes, n_features, n_features)

    z = random_state.randn(samples, n_features)
    allz = []
    for i in range(n_classes):
        allz.append(numpy.matmul(z, skews[i]))
    allz = numpy.array(allz)

    idx = numpy.arange(samples)
    x = mu[y] + allz[y[idx], idx, :]

    y = (
        OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1)).astype(numpy.float32)
        if one_hot_target
        else y.astype(numpy.int64)
    )
    x = tensor.to_backend(x.astype(numpy.float32))
    return x, tensor.to_backend(y)


def make_n_skewed_gaussian_datasets(
    samples: int = 300,
    n_classes: int = 3,
    n_features: int = 8,
    one_hot_target: bool = False,
    seed=0,
    n_datasets=10,
):
    gen = [
        make_high_skewed_gaussian(
            samples=samples,
            n_features=n_features,
            n_classes=n_classes,
            one_hot_target=one_hot_target,
            seed=seed + i,
        )
        for i in range(0, int(n_datasets * 2), 2)
    ]
    return tensor.concatenate([x[0] for x in gen]), tensor.concatenate([x[1] for x in gen])


def make_wines(seed=None, *args, **kwargs):
    """Load the wines dataset from sklearn with the appropriate format for program synthesis."""
    X, y = load_wine(return_X_y=True)
    x = tensor.to_backend(X.astype(numpy.float32))
    return x, tensor.to_backend(y)


def make_roll(n_classes=3, samples=256, seed=None, noise=0.0, *args, **kwargs):
    """Load the wines dataset from sklearn with the appropriate format for program synthesis."""
    X, y = make_swiss_roll(n_samples=samples, random_state=seed, noise=noise)
    bins = KBinsDiscretizer(n_bins=n_classes, encode="ordinal")
    y = bins.fit_transform(y.reshape(-1, 1)).astype(int)
    x = tensor.to_backend(X.astype(numpy.float32))
    return x, tensor.to_backend(y).flatten()


def make_rastrigin(n_classes=3, n_features=4, samples=256, seed=None, noise=0.0, *args, **kwargs):
    """Load the wines dataset from sklearn with the appropriate format for program synthesis."""
    X = numpy.random.uniform(-3, 3, size=(samples, n_features))
    X[-2] = 3.0
    X[-1] = 0.0
    bins = KBinsDiscretizer(n_bins=n_classes, encode="ordinal")
    y = rastrigin(X)
    y = bins.fit_transform(y.reshape(-1, 1)).astype(int)
    x = tensor.to_backend(X.astype(numpy.float32))
    return x, tensor.to_backend(y).flatten()


def make_sinusoids(n_classes=3, n_features=4, samples=256, seed=0, *args, **kwargs):
    random_state = numpy.random.RandomState(seed)
    coeffs = random_state.randn(1, n_features, 8, 3) * 2
    amps = random_state.randn(1, 8, 3) * 0.5
    coeffs[:, 2:, :, :] = 0

    phases = random_state.randn(1, 8, 3)

    x = random_state.randn(samples, n_features)
    x[:, 2:] = 0
    X = numpy.empty_like(x)
    for i in range(0, n_features, 2):
        X[:, i : i + 2] = x[:, :2]
    y = numpy.tanh(
        numpy.sum(
            amps
            * numpy.cos(
                numpy.sum(coeffs * x[:, :, numpy.newaxis, numpy.newaxis], axis=1) + phases
            ),
            axis=1,
        )
    ).argmax(-1)

    return X, y


def make_digits(seed=None, n_class: int = 10):
    """Load the digits dataset from sklearn with the appropriate format for program synthesis."""
    X, y = load_digits(n_class=n_class, return_X_y=True)
    x = tensor.to_backend(X.astype(numpy.float32))
    return x, tensor.to_backend(y)


def make_lunar_lander(seed=None, *args, **kwargs):
    """Load the wines dataset from sklearn with the appropriate format for program synthesis."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with open(os.path.join(path, "lunar_lander_x.pck"), "rb") as f:
        X = pickle.load(f)
    with open(os.path.join(path, "lunar_lander_y.pck"), "rb") as f:
        y = pickle.load(f)
    x = tensor.to_backend(X.astype(numpy.float32))
    return x, tensor.to_backend(y)
