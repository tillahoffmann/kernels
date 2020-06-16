import contextlib
import logging
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _safe_update(*dicts):
    """
    Join one or more dictionaries, ensuring there are no duplicate keys.
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                raise ValueError('duplicate key: %s' % key)
            result[key] = value
    return result


def expand_mapping(mapping):
    """
    Expand tuple keys in a mapping.

    Parameters
    ----------
    mapping : dict
        Mapping whose tuple keys to expand safely.

    Returns
    -------
    expanded : dict
        Mapping with expanded tuple keys.
    """
    tuple_keys = [keys for keys in mapping if isinstance(keys, tuple)]
    for keys in tuple_keys:
        value = mapping.pop(keys)
        mapping = _safe_update(mapping, {key: value for key in keys})
    return mapping


def expand_mappings(**mappings):
    """
    Expand tuples of keys in mappings.

    Parameters
    ----------
    mappings : dict
        Mappings keyed by attribute name, where each mapping is keyed by an attribute value to the
        target attribute value. If the attribute value is a tuple, the tuple will be expanded.

    Returns
    -------
    expanded : dict
        Expanded mapping of attributes.
    """
    return {attr: expand_mapping(mapping) for attr, mapping in mappings.items()}


def recode_values(x, **mappings):
    """
    Recode values according to a mapping.
    """
    mappings = expand_mappings(**mappings)
    return {key: mappings.get(key, {}).get(value, value) for key, value in x.items()}


def fill_values(x, *, fill_value=None, **mappings):
    """
    Fill values
    """
    return {key: fill_value if value in mappings.get(key, []) else value
            for key, value in x.items()}


def to_records(xs):
    """
    Convert a dictionary of values or a list of dictionaries to a record array.
    """
    return pd.DataFrame(xs).to_records(index=False)


class Dataset:
    """
    Base class for constructing datasets for survey inference.
    """
    def __init__(self, n, weighted):
        self.z = []
        self.egos = []
        self.alters = []
        self.invalid_egos = []
        self.invalid_alters = []
        self.pairs = []
        self.current_ego = None
        self.n = n
        self.weighted = weighted

    @staticmethod
    def get_attributes(row, mapping, *args):
        """
        Get attributes from a pandas row.
        """
        if args:
            mapping = {key: value % tuple(args) for key, value in mapping.items()}
        return {key: row[value] for key, value in mapping.items()}

    @contextlib.contextmanager
    def add_ego(self, ego):
        assert self.current_ego is None
        ego = self.recode(ego, True)

        # Skip straight away if the ego doesn't have a weight
        weight = ego.get('weight')
        if pd.isnull(weight) or weight <= 0:
            is_invalid = 'zero weight'
        else:
            is_invalid = self.is_invalid(ego, True)

        if is_invalid:
            ego['_invalid'] = is_invalid
            self.invalid_egos.append(ego)
            yield None
        else:
            idx = len(self.z)
            self.egos.append(idx)
            self.current_ego = idx
            self.z.append(ego)
            yield idx
            self.current_ego = None

    def add_alter(self, alter):
        assert self.current_ego is not None
        alter = self.recode(alter, False)

        is_invalid = self.is_invalid(alter, False)
        if is_invalid:
            alter['_invalid'] = is_invalid
            self.invalid_alters.append(alter)
        else:
            idx = len(self.z)
            self.alters.append(idx)
            self.z.append(alter)
            self.pairs.append((idx, self.current_ego))

    def is_invalid(self, x, ego):
        # Exclude all children
        age = x.get('age')
        if not pd.isnull(age) and age < 18:
            return 'age < 18'

        if all(map(pd.isnull, x.values())):
            return 'all values missing'
        if any(map(pd.isnull, x.values())):
            return 'some values missing'
        return False

    def recode(self, x, ego):
        return x

    def feature_map(self, x, y):
        raise NotImplementedError

    def load(self):
        z = to_records(self.z)
        for field, (dtype, _) in z.dtype.fields.items():
            # Report on object dtypes
            if dtype == object:
                unique = set(z[field])
                LOGGER.warning('field %s has object dtype with %d unique values', field,
                               len(unique))
                if len(unique) < 50:
                    LOGGER.warning('unique values for field %s: %s', field,
                                   ', '.join(map(str, unique)))

        return {
            'z': z,
            'egos': np.asarray(self.egos),
            'alters': np.asarray(self.alters),
            'pairs': np.asarray(self.pairs),
            'weighted': self.weighted,
            'n': self.n,
            'feature_map': self.feature_map,
            'invalid_egos': to_records(self.invalid_egos),
            'invalid_alters': to_records(self.invalid_alters),
        }
