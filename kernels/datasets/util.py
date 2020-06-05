import pandas as pd


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


def recode(x, **mappings):
    """
    Recode values according to a mapping.
    """
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
