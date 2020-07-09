"""
This module provides functions to generate synthetic datasets which can be used to test inference
algorithms.
"""

import numpy as np
from .util import get_args


def simulate_network_data(*, n, bias, link, s, seed, feature_map, theta_scale=1, **kwargs):
    """
    Generate ego network data using a soft random geometric graph (sRGG) whose nodes are sampled
    uniformly from the unit hypercube. The probability for two nodes to connect with one another is
    evaluated using a generalised linear model (GLM) with the specified link function and features.

    Parameters
    ----------
    n : int
        Population size.
    bias : array_like
        Bias term for the GLM.
    link : callable
        Link function for the GLM.
    s : int
        Size of seed sample.
    seed : int or None
        Random number generator seed.
    **kwargs : dict
        Additional keyword arguments associated with this simulation.

    Returns
    -------
    data : dict
        Container for the simulated data and the input configuration.
    """
    np.random.seed(seed)
    p = len(bias)

    while True:
        # Generate population-level features
        z = np.random.uniform(0, 1, (n, p - 1))
        x = feature_map(z[None, :], z[:, None])
        # Sample the coefficients and add the bias term
        theta = np.random.normal(0, 1, p) * theta_scale + bias
        # Calculate scores, probabilities, and simulate outcomes
        scores = np.sum(x * theta, axis=-1)
        probas = link(scores)
        np.fill_diagonal(probas, 0)
        y = np.random.uniform(0, 1, probas.shape) < probas
        edgelist = np.transpose(np.nonzero(y))

        # Sample egos and alters
        egos = np.random.choice(n, s, False)
        pairs = np.asarray([(i, j) for i, j in edgelist if j in egos])
        alters = np.unique(pairs[:, 0])

        if len(pairs) == 0:
            continue  # pragma: no cover

        return {
            # Simulated data
            'z': z,
            'x': x,
            'theta': theta,
            'y': y,
            'edgelist': edgelist,
            'pairs': pairs,
            'egos': egos,
            'alters': alters,
            'probas': probas,
            **get_args(),
        }


def simulate_data(*, n, bias, link, positive_sampling_weight, s, seed, theta_scale=1, **kwargs):
    """
    Generate standard case control test data.

    Parameters
    ----------
    n : int
        Population size.
    bias : array_like
        Bias term for the GLM.
    link : callable
        Link function for the GLM.
    positive_sampling_weight : float
        Weight with which cases should be over-sampled compared with non-cases.
    s : int
        Sample size.
    seed : int or None
        Random number generator seed.
    **kwargs : dict
        Additional keyword arguments associated with this simulation.

    Returns
    -------
    data : dict
        Container for the simulated data and the input configuration.
    """
    np.random.seed(seed)
    p = len(bias)

    while True:
        # Generate population-level features
        x = np.random.normal(0, 1, (n, p))
        x[..., 0] = 1
        # Sample the coefficients and add the bias term
        theta = np.random.normal(0, 1, p) * theta_scale + bias
        # Calculate scores, probabilities, and simulate outcomes
        scores = np.dot(x, theta)
        probas = link(scores)
        y = np.random.uniform(0, 1, probas.shape) < probas

        # Sample the observations
        weight = np.where(y, positive_sampling_weight, 1.0)
        observations = np.random.choice(n, s, False, weight / weight.sum())

        # Evaluate the marginal outcome distribution for the population and sample
        marginal = np.bincount(y, minlength=2) / y.shape[0]
        marginal_observed = np.bincount(y[observations], minlength=2) / observations.shape[0]

        if not np.any(y[observations]):
            continue  # pragma: no cover

        data = {
            # Simulated data
            'x': x,
            'y': y,
            'theta': theta,
            'observations': observations,
            'marginal': marginal,
            'marginal_observed': marginal_observed,
            'x_observed': x[observations],
            'y_observed': y[observations],
            **get_args(),
        }
        return data
