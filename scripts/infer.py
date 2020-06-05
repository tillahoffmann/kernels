import argparse
from atomicwrites import atomic_write
import functools as ft
import hashlib
import kernels
import logging
import numpy as np
import numpy.lib.recfunctions
import os
import pickle
import re
from scipy import optimize
from scipy import special
from scipy import stats


RANGE_PATTERN = re.compile(r'(\d+):(\d+)')


def _parse_seed(value):
    values = []
    parts = value.split(',')
    for part in parts:
        part = part.strip()
        match = RANGE_PATTERN.match(part)
        if match:
            start, end = map(int, match.groups())
            values.extend(range(start, end))
        else:
            values.append(int(part))
    return values


# Boilerplate --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['alp', 'gss', 'synthetic'],
                    help='dataset to run inference on')
parser.add_argument('--num_samples', '-n', type=int, default=10000,
                    help='number of posterior samples')
parser.add_argument('--log-level', choices=['info', 'debug'], default='info')
parser.add_argument('--filename', '-f', help='filename to store results')
parser.add_argument('--seed', '-s', type=_parse_seed, help='random number generator seed')
parser.add_argument('--prior', '-p', choices=['flat', 'cauchy'], default='cauchy',
                    help='prior for kernel parameters')
parser.add_argument('--force', '-B', action='store_true', default=False,
                    help='reevaluate inference even if results are present')
args = parser.parse_args()

# General setup
logging.basicConfig(level=args.log_level.upper())


# Iterate over seeds
data = None
for seed in args.seed or [None]:
    arghash = hashlib.sha1()
    config = dict(vars(args))
    config['seed'] = seed
    config.pop('force', None)
    for key, value in sorted(config.items()):
        arghash.update(key.encode())
        arghash.update(repr(value).encode())
    arghash = arghash.hexdigest()

    logging.info('using configuration with hash %s: %s', arghash, config)

    filename = args.filename or f'workspace/{args.dataset}/{arghash}.pkl'
    if os.path.isfile(filename) and not args.force and seed:
        logging.info('skipping %s as results are already present', filename)
        continue

    np.random.seed(seed)
    logging.info('seeded with %s', seed)

    # Load the data --------------------------------------------------------------------------------

    if args.dataset == 'synthetic':
        configuration = {
            'n': 2000,
            'bias': [-7, 0, 0],
            'link': special.expit,
            'linkln': kernels.expitln,
            's': 100,
            'seed': seed,
            'feature_map': kernels.l1_feature_map
        }
        data = kernels.simulate_network_data(**configuration)
    elif args.dataset == 'gss':
        data = kernels.datasets.load_general_social_survey('data/GSS2004.dta')
    else:
        raise KeyError(args.dataset)

    # Evaluate an estimate of the prevalence in the population
    k = len(data['pairs']) / len(data['egos'])
    prevalence = k / (data['n'] - 1)

    logging.info('loaded dataset `%s`', args.dataset)
    logging.info('number of respondents: %d', len(data['egos']))
    logging.info('number of edges: %d', len(data['pairs']))
    logging.info('mean degree: %f', k)

    # Construct the log-likelihood and log-posterior -----------------------------------------------

    # Get the features for cases
    i1, j1 = np.transpose(data['pairs'])
    z = data['z']
    x_cases = data['feature_map'](z[i1], z[j1])

    # Sample controls and get their features
    i0, j0 = kernels.sample_controls(np.asarray(data['egos']), 3 * len(data['pairs']))
    x_controls = data['feature_map'](z[i0], z[j0])

    # Concatenate features and construct indicator variables
    x_observed = np.concatenate([x_cases, x_controls])
    y_observed = np.concatenate([np.ones(len(x_cases)), np.zeros(len(x_controls))]).astype(bool)

    # Construct a weight vector
    key = data.get('weights')
    if key is None:
        weights = None
    else:
        # Winsorize the weights
        ego_weights = z[key]
        p95 = np.nanpercentile(ego_weights, 95)
        ego_weights = np.where(ego_weights < p95, ego_weights, p95)
        # Construct weights as described in the appendix
        weights0 = ego_weights[i0] * ego_weights[j0]
        weights1 = ego_weights[j1]
        weights = np.concatenate([weights0, weights1])
        assert all(np.isfinite(weights))

    # Generate feature names for the synthetic case
    if x_controls.dtype.fields is None:
        nfeatures = x_controls.shape[-1]
        feature_names = ['f%d' % i for i in range(nfeatures)]
        offsets = np.zeros(nfeatures)
        scales = np.ones(nfeatures)
    # Extract feature names and standardise the features
    else:
        feature_names = []
        offsets = []
        scales = []
        for field, (dtype, _) in x_controls.dtype.fields.items():
            feature_names.append(field)
            if field == 'bias':
                offset = 0
                scale = 1
            else:
                offset = np.mean(x_controls[field])
                scale = 1 if dtype == bool else (2 * np.std(x_controls[field]))
            offsets.append(offset)
            scales.append(scale)
        offsets = np.asarray(offsets)
        scales = np.asarray(scales)
        x_observed = np.lib.recfunctions.structured_to_unstructured(x_observed)

    logging.info("feature offset: %s", dict(zip(feature_names, offsets)))
    logging.info("feature scale: %s", dict(zip(feature_names, scales)))

    # Standardise the features
    x_observed = (x_observed - offsets) / scales

    logging.info("mean standardised features for controls: %s",
                 dict(zip(feature_names, x_observed[~y_observed].mean(axis=0))))
    logging.info("mean standardised features for cases: %s",
                 dict(zip(feature_names, x_observed[y_observed].mean(axis=0))))

    log_likelihood = ft.partial(
        kernels.evaluate_case_control_log_likelihood,
        x=x_observed,
        y=y_observed,
        prevalence=prevalence,
        linkln=data.get('linkln', kernels.expitln),
        weights=weights,
    )

    def log_posterior(theta, **kwargs):
        """
        Evaluate the unnormalised log posterior, i.e. log prior plus log likelihood.
        """
        if args.prior == 'cauchy':
            scale = 2.5 * np.ones_like(theta)
            scale[0] = 10
            log_prior = -np.log1p(np.square(theta / scale)).sum()
        elif args.prior == 'flat':
            log_prior = 0
        else:
            raise KeyError(args.prior)
        return log_likelihood(theta, **kwargs) + log_prior

    # Optimise the log-posterior -------------------------------------------------------------------

    # Choose initial conditions based on logit of overall prevalence (and a bit of noise)
    x0 = np.random.normal(0, 1, x_observed.shape[1])
    x0[0] = special.logit(prevalence)
    logging.info('starting posterior maximisation from initial conditions: %s',
                 dict(zip(feature_names, x0)))

    # We optimise the log posterior averaged over observations rather than summed to avoid problems
    # with the numerical optimisation (see https://stackoverflow.com/a/54446479/1150961 for details)
    result = optimize.minimize(lambda x: -log_posterior(x) / len(x_observed), x0)
    assert result.success, result.message
    cov = result.hess_inv / len(x_observed)
    logging.info('maximised posterior in %d function evaluations', result.nfev)
    logging.info('MAP estimate: %s', dict(zip(feature_names, result.x)))
    logging.info('approximate marginal std: %s', dict(zip(feature_names, np.sqrt(np.diag(cov)))))

    # Draw samples from the log-posterior ----------------------------------------------------------
    # Use the inverse Hessian from the optimisation to construct an approximate "optimal" proposal
    # covariance following A. Gelman, G. O. Roberts, W. R. Gilks. "Efficient Metropolis jumping
    # rules". (1996)

    proposal_cov = 2.4 ** 2 * cov / len(result.x)
    xs, values = kernels.sample(log_posterior, result.x, proposal_cov, args.num_samples)
    acceptance = kernels.evaluate_acceptance(values)
    logging.info('obtained %d posterior samples with acceptance %.3f', args.num_samples, acceptance)
    logging.info('posterior mean: %s', dict(zip(feature_names, np.mean(xs, axis=0))))
    logging.info('posterior std: %s', dict(zip(feature_names, np.std(xs, axis=0))))
    if 'theta' in data:
        logging.info('true values: %s', dict(zip(feature_names, data['theta'])))
        residuals = np.mean(xs, axis=0) - data['theta']
        logging.info('z-scores: %s', dict(zip(feature_names, residuals / np.std(xs, axis=0))))
        cov = np.cov(xs.T)
        chi2 = residuals.dot(np.linalg.inv(cov)).dot(residuals)
        pval = 1 - stats.chi2(len(cov)).cdf(chi2)
        logging.info('chi2 for %d dof: %f; p-val: %f', len(cov), chi2, pval)

    # Package the data and results and save them ---------------------------------------------------
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with atomic_write(filename, mode='wb', overwrite=True) as fp:
        pickle.dump({
            'arghash': arghash,
            'args': config,
            'data': data,
            'result': result,
            'samples': {
                'xs': xs,
                'values': values,
            },
            'feature_names': feature_names,
            'offsets': offsets,
            'scales': scales,
        }, fp)

    logging.info('saved output to %s', filename)
