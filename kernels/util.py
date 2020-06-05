"""
``.util``
---------

The ``.util`` module provides utility functions, including evaluation of the case-control log
likelihood function and a Metropolis-Hastings Monte Carlo sampler.
"""

import functools as ft
import gc
import inspect
import numpy as np
from scipy import special


def negate(func):
    """
    Decorator to negate a function.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs):
        return - func(*args, **kwargs)
    return _wrapper


def expitln(x):
    """
    Evaluate the natural logarithm of the logistic function.
    """
    return -np.log1p(np.exp(-x))


def sample(log_dist, x, cov, num, args=None, kwargs=None):
    """
    Draw samples from a target distribution using a Markov chain Monte Carlo algorithm with Gaussian
    proposal distribution.

    Parameters
    ----------
    log_dist : callable
        Natural logarithm of the target distribution.
    x : array_like
        Initial parameters for the Markov chain.
    cov : array_like
        Covariance of proposal distribution.
    num : int
        Number of samples to draw.
    args : iterable
        Additional positional arguments to pass to `log_dist`.
    kwargs : dict
        Additional keyword arguments to pass to `log_dist`.

    Returns
    -------
    xs : np.ndarray
        `num` samples drawn from `log_dist`.
    values : np.ndarray
        `num` evaluations of `log_dist` at `samples`.
    """
    # Validate the inputs
    x = np.atleast_1d(x)
    if np.ndim(x) > 1:
        raise ValueError("`x` must have rank less than 2.")
    k, = x.shape

    if np.ndim(cov) < 2:
        cov = np.eye(x.shape[0]) * cov
    if np.ndim(cov) > 2:
        raise ValueError("`cov` must have rank less than 3.")
    if np.shape(cov) != (k, k):
        raise ValueError("Shape of `cov` does not match `x`.")

    value = -np.inf
    args = args or []
    kwargs = kwargs or {}

    xs = []
    values = []

    for i in range(num):
        candidate = np.random.multivariate_normal(x, cov)
        candidate_value = np.sum(log_dist(candidate, *args, **kwargs))
        if np.log(np.random.uniform(0, 1)) < candidate_value - value:
            x = candidate
            value = candidate_value
        xs.append(x)
        values.append(value)

    return np.asarray(xs), np.asarray(values)


def evaluate_acceptance(values):
    """
    Evaluate the acceptance probability given a sequence of values.

    Parameters
    ----------
    values : aray_like
        Values sampled using a Markov chain Monte Carlo algorithm.

    Returns
    -------
    acceptance : float
        Fraction of successive values that changed, i.e. the fraction of proposals accepted.
    """
    if np.ndim(values) != 1:
        raise ValueError("`values` must be a vector.")
    return np.mean(np.diff(values) != 0)


def evaluate_case_control_log_likelihood(theta, x, y, *, prevalence=None, ratio=None, linkln=None,
                                         aggregate=True, weights=None, _return_locals=False):
    """
    Evaluate the case-control log-likelihood for binary data under a generalised linear model (GLM)
    assuming a known population prevalence.

    .. note::
       Exactly one of ``prevalence`` or ``ratio`` must be specified.

    Parameters
    ----------
    theta : array_like with shape (p,)
        GLM parameters.
    x : array_like with shape (n, p)
        Features predicting the binary outcome.
    y : array_like with shape (n,)
        Binary outcomes.
    prevalence : float or array_like with shape (2,)
        Prevalence of cases in the general population.
    ratio : array_like with shape (2,) or (n, 2)
        Ratio of prevalence in the sample to prevalence in the population.
    linkln : callable
        Log link function (defaults to `expitln`).
    aggregate : str or boolean
        Method used to aggregate contributions to the log likelihood. If false-y, individual values
        for each data point are returned.
    weights : np.ndarray
        Weights associated with each element in ``x``. When weights are provided, ``weights`` must
        be ``True`` (the default).
    _return_locals : bool
        Whether to return all local variables for debugging purposes.

    Returns
    -------
    log_likelihood : np.ndarray
        Log-likelihood under the GLM (or pointwise contributions if `aggregate` is false-ish).
    """
    # Validate the inputs
    y = np.asarray(y).astype(int)
    if np.ndim(y) != 1:
        raise ValueError("`y` must be a vector.")
    n, = np.shape(y)
    if np.ndim(theta) != 1:
        raise ValueError("`theta` must be a vector.")
    p, = np.shape(theta)
    if np.shape(x) != (n, p):
        raise ValueError("Shape `x` does not match `y` or `theta`.")
    linkln = linkln or expitln

    if (prevalence is None) == (ratio is None):
        raise ValueError("Exactly one of `prevalence` or `ratio` must be specified.")
    if ratio is None:
        if np.ndim(prevalence) == 0:
            prevalence = np.asarray([1 - prevalence, prevalence])
        if np.shape(prevalence) != (2,):
            raise ValueError("`prevalence` must be a scalar or length-two vector.")

        # Evaluate the log prevalence ratio in the sample and the population
        marginal_observed = np.bincount(y, minlength=2) / y.shape[0]
        ratioln = np.log(marginal_observed / prevalence)
    else:
        if np.shape(ratio) != (2,) and np.shape(ratio) != (n, 2):
            raise ValueError("`ratio` must be a length-two vector or `(n, 2)` matrix.")
        ratioln = np.log(ratio)

    # Evaluate the scores prior to application of the link function
    scores = np.dot(x, theta)
    # Evaluate the log probability of the observation under an ignorable data collection
    probasln = linkln(scores * (2 * y - 1))
    # Evaluate the log adjustment factor
    adjustmentln = special.logsumexp(ratioln + linkln(scores[:, None] * [-1, 1]), axis=1)
    # Evaluate the log likelihood given the non-ignorable data collection
    log_likelihood = probasln - adjustmentln

    if weights is not None:
        if aggregate is not True:
            raise ValueError("`aggregate` must be `True` if weights are used.")
        log_likelihood = weights.dot(log_likelihood)
    elif aggregate:
        if aggregate is True:
            aggregate = 'sum'
        log_likelihood = getattr(np, aggregate)(log_likelihood)

    if _return_locals:  # pragma: no cover
        return locals()
    else:
        return log_likelihood


def exrange(x, factor=0.2, axis=None):
    """
    Evaluate an expanded range.
    """
    xmin = np.min(x, axis=axis)
    xmax = np.max(x, axis=axis)
    rng = xmax - xmin
    return xmin - factor * rng, xmax + factor * rng


def add_bias_feature(features):
    """
    Add a bias feature.
    """
    shape = np.shape(features)[:-1] + (1,)
    return np.concatenate([np.ones(shape), features], axis=-1)


def l1_feature_map(x, y, offset='hypercube', scale='hypercube'):
    """
    Evaluate L1 features, including a bias term and feature standardisation.

    Parameters
    ----------
    x : np.ndarray
        Attributes of alters.
    y : np.ndarray
        Attributes of egos.
    offset : np.ndarray or str
        Offset to subtract from features. If `hypercube`, assume attributes are sampled uniformly
        from a hypercube.
    scale : np.ndarray or str
        Scale to divide features by. If `hypercube`, assume attributes are sampled uniformly from a
        hypercube.

    Returns
    -------
    features : np.ndarray
        L1 features.
    """
    if isinstance(offset, str):
        if offset == 'hypercube':
            # Subtract the mean
            offset = 1 / 3
        else:
            raise ValueError(offset)

    if isinstance(scale, str):
        if scale == 'hypercube':
            # Divide by twice the standard deviation as outlined by
            # A. Gelman. "Scaling regression inputs by dividing by two standard deviations". (2008)
            scale = 2 / (3 * np.sqrt(2))
        else:
            raise ValueError(offset)

    return add_bias_feature((np.abs(x - y) - offset) / scale)


def get_args():
    """
    Retrieves the arguments of the calling function.
    """
    # Get the parent frame and obtain the calling function
    frame = inspect.currentframe().f_back
    func = None
    for referrer in gc.get_referrers(frame.f_code):
        if callable(referrer) and referrer.__code__ is frame.f_code:
            func = referrer
    assert func, "could not resolve function"

    # Extract the arguments from the frame
    argspec = inspect.getfullargspec(func)
    args = {
        key: frame.f_locals[key] for key in argspec.args + argspec.kwonlyargs
    }
    if argspec.varargs:
        args['*varargs'] = frame.f_locals[argspec.varargs]
    if argspec.varkw:
        args.update(frame.f_locals[argspec.varkw])

    return args


def square_to_condensed(i, j, n):
    """
    Convert indices of a square distance matrix to the index of a condensed array index.

    Parameters
    ----------
    i : np.ndarray
        Row index.
    j : np.ndarray
        Column index.
    n : int
        Number of rows and columns.

    Returns
    -------
    c : np.ndarray
        Condensed index.
    """
    # Ensure correct ordering
    i, j = np.minimum(i, j), np.maximum(i, j)
    # Evaluate the index
    return i * n + j - i * (i + 1) // 2 - i - 1


def condensed_to_square(c, n):
    """
    Convert the index of a condensed array to indices of a square distance matrix.

    Parameters
    ----------
    c : np.ndarray
        Condensed index.
    n : int
        Number of rows and columns.

    Returns
    -------
    i : np.ndarray
        Row index.
    j : np.ndarray
        Column index.
    """
    dtype = c.dtype
    i = np.ceil((2 * n - 1 - np.sqrt(4 * n ** 2 - 4 * n - 8 * c - 7)) / 2 - 1)
    j = n - ((i + 1) * (n - i - 2) + ((i + 1) * (i + 2)) // 2) + c
    np.testing.assert_array_less(i, n)
    np.testing.assert_array_less(j, n)
    return i.astype(dtype), j.astype(dtype)


def sample_controls(egos, n):
    """
    Sample distinct pairs of controls.

    Parameters
    ----------
    egos : np.ndarray
        Egos to sample from.
    n : int
        Number of distinct pairs to sample.

    Returns
    -------
    i : np.ndarray
        Sampled alters.
    j : np.ndarray
        Sampled egos.
    """
    # Check we can sample as many pairs as requested
    nmax = len(egos) * (len(egos) - 1) // 2
    # Sample indices from the "flattened" pairs (cf. scipy.spatial.distance.squareform)
    idx = np.random.choice(nmax, n, False)
    # Convert to indices
    i, j = condensed_to_square(idx, len(egos))
    return egos[i], egos[j]
