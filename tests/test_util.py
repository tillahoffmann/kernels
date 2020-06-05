import numpy as np
import kernels
import pytest
from scipy.spatial import distance


@pytest.mark.parametrize('n', (10, 17, 99))
def test_index_conversion(n):
    # Evaluate condensed an square distance matrix
    x = np.random.normal(0, 1, (n, 2))
    y = distance.pdist(x)
    d = distance.squareform(y)

    # Sample indices and filter them
    i, j = np.random.randint(n, size=(2, 100))
    f = i != j
    i, j = i[f], j[f]

    # Evaluate condensed index
    c = kernels.square_to_condensed(i, j, n)
    np.testing.assert_array_equal(y[c], d[i, j])

    # Make sure we can recover the original indices
    i2, j2 = kernels.condensed_to_square(c, n)
    np.testing.assert_array_equal(np.minimum(i, j), i2)
    np.testing.assert_array_equal(np.maximum(i, j), j2)


def test_negate():
    assert kernels.negate(lambda: 4)() == -4


def test_l1_feature_map():
    n = 10000
    x, y = np.random.uniform(0, 1, (2, n, 1))
    features = kernels.l1_feature_map(x, y)
    assert features.shape == (n, 2)
    np.testing.assert_allclose(features[..., 0], 1)
    mean = np.mean(features[..., 1])
    std = np.std(features[:, 1])
    z = mean / (std / np.sqrt(n))
    # Assert that the mean is close to zero (this test may fail on occasion if the rng is not
    # seeded)
    assert abs(z) < 3
    # Assert that the standard deviation is close to .5 (after standardisation following Gelman)
    assert abs(std - 0.5) < 0.01


def test_l1_feature_map_invalid_arguments():
    with pytest.raises(ValueError):
        kernels.l1_feature_map(None, None, offset='invalid')
    with pytest.raises(ValueError):
        kernels.l1_feature_map(None, None, scale='invalid')


def test_case_control_invalid_arguments():
    n, p = 100, 3
    x = np.random.normal(0, 1, (n, p))
    theta = np.random.normal(0, 1, p)
    y = np.random.normal(0, 1, n) > 0

    with pytest.raises(ValueError):
        kernels.evaluate_case_control_log_likelihood(theta, x, 0)
    with pytest.raises(ValueError):
        kernels.evaluate_case_control_log_likelihood(0, x, y)
    with pytest.raises(ValueError):
        kernels.evaluate_case_control_log_likelihood(theta, 0, y)
    with pytest.raises(ValueError):
        kernels.evaluate_case_control_log_likelihood(theta, x, y)
    with pytest.raises(ValueError):
        kernels.evaluate_case_control_log_likelihood(theta, x, y, prevalence=np.empty(3))


def test_sample():
    xs, values = kernels.sample(lambda x: - x ** 2 / 2, 0, 1, 10000)
    assert abs(np.mean(xs)) < 0.1
    assert abs(np.std(xs) - 1) < 0.1


def test_sample_invalid_arguments():
    def _log_dist(_):
        return 0
    x = 0
    cov = 1
    num = 10

    with pytest.raises(ValueError):
        kernels.sample(_log_dist, np.empty((1, 1)), cov, num)
    with pytest.raises(ValueError):
        kernels.sample(_log_dist, x, np.empty((1, 1, 1)), num)
    with pytest.raises(ValueError):
        kernels.sample(_log_dist, x, np.empty((1, 2)), num)


def test_get_args():
    def _func(a, *args, b, **kwargs):
        return kernels.get_args()

    args = _func(3, 4, 5, b=np.pi, foo='bar')
    assert args == {
        'a': 3,
        '*varargs': (4, 5),
        'b': np.pi,
        'foo': 'bar',
    }
