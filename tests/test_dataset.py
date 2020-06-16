from kernels.datasets import dataset as util
import numpy as np
import pytest


def test_expand_mapping():
    expanded = util.expand_mapping({
        1: 3,
        (3, 4): 2
    })
    assert expanded == {1: 3, 3: 2, 4: 2}


def test_expand_mapping_duplicate():
    with pytest.raises(ValueError):
        util.expand_mapping({1: 2, (1, 3): 'a'})
    with pytest.raises(ValueError):
        util.expand_mapping({(1, 2): 'a', (2, 3): 'b'})


def test_expand_mappings():
    expanded = util.expand_mappings(**{
        'attr1': {
            3: 'a',
        },
        'attr2': {
            (3, 4): 'a',
            5: 'b'
        }
    })

    assert expanded == {
        'attr1': {
            3: 'a',
        },
        'attr2': {
            3: 'a',
            4: 'a',
            5: 'b',
        }
    }


def test_recode_values():
    x = {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    mappings = {
        'a': {
            1: -1
        },
        'b': {
            5: None,
        }
    }
    assert util.recode_values(x, **mappings) == {
        'a': -1,
        'b': 2,
        'c': 3,
    }


def test_fill_values():
    x = {
        'a': 19,
        'b': 7,
    }
    assert util.fill_values(x, fill_value='7', **{'a': [19]}) == {'a': '7', 'b': 7}


def test_to_records_list():
    rec = util.to_records([
        {'a': 1, 'b': True},
        {'a': 2, 'b': False},
    ])
    np.testing.assert_array_equal(rec['a'], [1, 2])


def test_to_records_dict():
    rec = util.to_records({
        'a': [1, 2],
        'b': [True, False],
    })
    np.testing.assert_array_equal(rec['b'], [True, False])
