import functools as ft
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from scipy import special
from scipy import stats
import kernels


matplotlib.use('agg')


def _plot_inference(theta, xs, result, filename=None):
    """
    Generate an inference plot.
    """
    # Compute the chi2 and pvalue assuming a Gaussian posterior
    residuals = xs.mean(axis=0) - theta
    cov = np.cov(xs.T)
    chi2 = residuals.dot(np.linalg.inv(cov)).dot(residuals)
    pval = 1 - stats.chi2.cdf(chi2, len(residuals))

    # Generate plot to inspect the inference
    fig, ax = plt.subplots()
    for i in range(theta.shape[0]):
        color = f'C{i}'
        x = xs[:, i]
        linx = np.linspace(*kernels.exrange(x))
        ax.plot(linx, stats.gaussian_kde(x)(linx), color=color)
        ax.axvline(theta[i], color=color)
        ax.axvline(result.x[i], color=color, ls='--')

    text = f'''
    chi2 = {chi2:.3f}
    pval = {pval:.3f}
    acceptance = {kernels.evaluate_acceptance(xs[:, 0]):.3f}
    '''
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, ha='left', va='top')
    fig.tight_layout()

    # Save the figure before testing to ensure we can inspect it
    if filename:
        fig.savefig(filename)

    # Assert that the inference is vaguely sensible (this test may fail on occasion if rng is not
    # seeded)
    assert pval > .05, "inference may not be consistent with seeded parameters"
    return fig


def test_case_control():
    configuration = {
        'n': 20000,
        'bias': [-3, 0, 0],
        'positive_sampling_weight': 10,
        's': 200,
        'seed': 3,
        'link': special.expit,
        'linkln': kernels.expitln,
    }

    data = kernels.simulate_data(**configuration)
    log_likelihood = ft.partial(
        kernels.evaluate_case_control_log_likelihood,
        x=data['x_observed'],
        y=data['y_observed'],
        prevalence=data['marginal'],
        linkln=data['linkln'],
    )

    # Find the maximum likelihood estimate
    theta = data['theta']
    x0 = np.random.normal(size=theta.shape)
    # Use mean aggregation to avoid https://stackoverflow.com/a/54446479/1150961
    result = optimize.minimize(lambda x: -log_likelihood(x, aggregate='mean'), x0)
    assert result.success, 'optimization failed'

    # Draw samples and plot them
    cov = 2.4 ** 2 * result.hess_inv / (len(data['x_observed']) * len(theta))
    xs, values = kernels.sample(log_likelihood, result.x, cov, 5000)
    _plot_inference(theta, xs, result, filename='tests/~case_control.png')


def test_network_inference():
    # Sample data
    configuration = {
        'n': 2000,
        'bias': [-7, 0, 0],
        'link': special.expit,
        'linkln': kernels.expitln,
        's': 100,
        'seed': 0,
        'feature_map': kernels.l1_feature_map
    }
    data = kernels.simulate_network_data(**configuration)

    # Evaluate an estimate of the prevalence in the population
    k = len(data['pairs']) / data['s']
    prevalence = k / (data['n'] - 1)

    # Get the features for cases
    i1, j1 = data['pairs'].T
    z = data['z']
    x_cases = data['feature_map'](z[i1], z[j1])

    # Sample controls and get their features
    i0, j0 = kernels.sample_controls(data['egos'], 3 * len(data['pairs']))
    x_controls = data['feature_map'](z[i0], z[j0])

    # Concatenate features and construct indicator variables
    x_observed = np.concatenate([x_cases, x_controls])
    y_observed = np.concatenate([np.ones(len(x_cases)), np.zeros(len(x_controls))])

    log_likelihood = ft.partial(
        kernels.evaluate_case_control_log_likelihood,
        x=x_observed,
        y=y_observed,
        prevalence=prevalence,
        linkln=data['linkln'],
    )

    # Find the maximum likelihood estimate
    theta = data['theta']
    x0 = np.random.normal(size=theta.shape)
    # Use mean aggregation to avoid https://stackoverflow.com/a/54446479/1150961
    result = optimize.minimize(lambda x: -log_likelihood(x, aggregate='mean'), x0)
    assert result.success, 'optimization failed'

    # Draw samples and plot them
    cov = 2.4 ** 2 * result.hess_inv / (len(x_observed) * len(theta))
    xs, values = kernels.sample(log_likelihood, result.x, cov, 5000)
    _plot_inference(theta, xs, result, filename='tests/~network_inference.png')
