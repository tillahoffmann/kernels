import functools as ft
import logging
import numpy as np
import pandas as pd
from . import util

ETHNICITY_MAP = {
    'white': (1, 2, 3, 4, 5, 6, 7),
    'asian': (9, 10, 11, 12, 13, 7),
    'black': (14, 15, 16, 5, 6),
    'arab': (17,),
    'other': (8, 97),
}
LOGGER = logging.getLogger(__name__)


def understanding_society_feature_map(x, y, sampled_distances):
    """
    Evaluate features for the Understanding Society Survey.
    """
    # Aggregate in the same fashion as Understanding Society
    proba, _ = np.histogram(sampled_distances, bins=[0, 1, 5, 50, 1e6])
    proba = proba / np.sum(proba)
    missing = np.isnan(x['distance'])
    sampled = 1 + np.random.choice(len(proba), len(x), p=proba)
    distance = np.where(missing, sampled, x['distance'])

    # Get the ethnicities
    keys = ['ethnicity_%s' % key for key in ETHNICITY_MAP]
    ethnicity = 0.5 * sum(abs(x[key] - y[key]) for key in keys)

    return util.to_records({
        'bias': np.ones(x.shape[0]),
        'sex': x['sex'] != y['sex'],
        'age': np.abs(x['age'] - y['age']),
        'occupation': x['occupation'] != y['occupation'],
        'ethnicity': ethnicity,
        'distance': distance,
    })


def _recode_ethnicity(x):
    ethnicity = x.pop('ethnicity')
    ethnicities = {key: ethnicity in value for key, value in ETHNICITY_MAP.items()}
    norm = sum(ethnicities.values())
    assert norm, "no ethnicity recovered from code %s" % ethnicity
    x.update({'ethnicity_%s' % key: value / norm for key, value in ethnicities.items()})
    return x


def load_understanding_society_survey(filename, code, distance_filename):
    """
    Load the dataset from the Understanding Society Survey.

    Parameters
    ----------
    filename : str
        Filename from which to load data.
    code : str
        Prefix code for variable names across different waves of the survey.
    distance_filename : str
        Filename from which to load the reference distance distribution.

    Notes
    -----
    This section discusses coding for both respondents and nominees and details the steps taken to
    harmonise codings.

    ``sex`` is coded as follows for
    `respondents <https://www.understandingsociety.ac.uk/documentation/mainstage/
    dataset-documentation/wave/3/datafile/c_indresp/variable/c_sex>`__ and
    `nominees <https://www.understandingsociety.ac.uk/documentation/mainstage/
    dataset-documentation/wave/3/datafile/c_indresp/variable/c_netsx_1>`__.

    +---------------+--------------+------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value | Respondent Code |
    +===============+==============+==================+=================+
    | male          | 1            | male             | 1               |
    +---------------+--------------+------------------+-----------------+
    | female        | 2            | female           | 2               |
    +---------------+--------------+------------------+-----------------+
    | don't know    | -1           | ---                                |
    +---------------+--------------+------------------+-----------------+
    | refusal       | -2           | ---                                |
    +---------------+--------------+------------------+-----------------+
    | proxy         | -7           | ---                                |
    +---------------+--------------+------------------+-----------------+
    | inapplicable  | -8           | ---                                |
    +---------------+--------------+------------------+-----------------+
    | n/a for IEMB  | -10          | ---                                |
    +---------------+--------------+------------------+-----------------+

    ``age`` is coded as a numerical value for `respondents <https://www.understandingsociety.ac.uk/
    documentation/mainstage/dataset-documentation/wave/3/datafile/c_indresp/variable/c_dvage>`__ and
    `nominees <https://www.understandingsociety.ac.uk/documentation/mainstage/dataset-documentation/
    wave/3/datafile/c_indresp/variable/c_netag_1>`__.

    +------------------+--------------+------------------+-----------------+
    | Nominee Value    | Nominee Code | Respondent Value | Respondent Code |
    +==================+==============+==================+=================+
    | numeric value                                                        |
    +------------------+--------------+------------------+-----------------+
    | invalid as above | < 0          | ---                                |
    +------------------+--------------+------------------+-----------------+

    ``occupation`` is coded as follows for `respondents <https://www.understandingsociety.ac.uk/
    documentation/mainstage/dataset-documentation/wave/3/datafile/c_indresp/variable/c_jbstat>`__
    and `nominees <https://www.understandingsociety.ac.uk/documentation/mainstage/
    dataset-documentation/wave/3/datafile/c_indresp/variable/c_netjb_1)>`__.

    +----------------------+--------------+----------------------------+-----------------+
    | Nominee Value        | Nominee Code | Respondent Value           | Respondent Code |
    +======================+==============+============================+=================+
    | full- or part-time   | 1, 2         | self employed              | 1               |
    | employment           |              +----------------------------+-----------------+
    |                      |              | in paid employment         | 2               |
    |                      |              +----------------------------+-----------------+
    |                      |              | on maternity leave         | 5               |
    |                      |              +----------------------------+-----------------+
    |                      |              | unpaid family business     | 10              |
    +----------------------+--------------+----------------------------+-----------------+
    | unemployed           | 3            | unemployed                 | 3               |
    |                      |              +----------------------------+-----------------+
    |                      |              | long-term sick or disabled | 8               |
    +----------------------+--------------+----------------------------+-----------------+
    | full-time education  | 4            | full-time student          | 7               |
    |                      |              +----------------------------+-----------------+
    |                      |              | government training scheme | 9               |
    |                      |              +----------------------------+-----------------+
    |                      |              | on apprenticeship          | 11              |
    +----------------------+--------------+----------------------------+-----------------+
    | full-time housework  | 5            | family care                | 6               |
    +----------------------+--------------+----------------------------+-----------------+
    | fully retired        | 6            | retired                    | 4               |
    +----------------------+--------------+----------------------------+-----------------+
    | ---                                 | doing something else       | 97              |
    +----------------------+--------------+----------------------------+-----------------+
    | invalid as above     | < 0          |                            |                 |
    +----------------------+--------------+----------------------------+-----------------+

    ``ethnicity`` is coded consistently for `respondents <https://www.understandingsociety.ac.uk/
    documentation/mainstage/dataset-documentation/wave/3/datafile/c_indresp/variable/c_racel_dv>`__
    and `nominees <https://www.understandingsociety.ac.uk/documentation/mainstage/
    dataset-documentation/wave/3/datafile/c_indresp/variable/c_netet_1>`__. We aggregate detailed
    codings to more generic non-exclusive categories, i.e. an individual may identify with more than
    one.

    +----------------------+---------------------------------------+-----------------+
    | Category             | Value                                 | Code            |
    +======================+=======================================+=================+
    | White                | British, English, Scottish, Welsh, NI | 1               |
    |                      +---------------------------------------+-----------------+
    |                      | Irish                                 | 2               |
    |                      +---------------------------------------+-----------------+
    |                      | Gypsy or Irish traveller              | 3               |
    |                      +---------------------------------------+-----------------+
    |                      | Other White background                | 4               |
    |                      +---------------------------------------+-----------------+
    |                      | White and Black Caribbean             | 5               |
    |                      +---------------------------------------+-----------------+
    |                      | White and Black African               | 6               |
    |                      +---------------------------------------+-----------------+
    |                      | White and Asian                       | 7               |
    +----------------------+---------------------------------------+-----------------+
    | Asian                | Indian                                | 9               |
    |                      +---------------------------------------+-----------------+
    |                      | Pakistani                             | 10              |
    |                      +---------------------------------------+-----------------+
    |                      | Bangladeshi                           | 11              |
    |                      +---------------------------------------+-----------------+
    |                      | Chinese                               | 12              |
    |                      +---------------------------------------+-----------------+
    |                      | Other Asian background                | 13              |
    |                      +---------------------------------------+-----------------+
    |                      | White and Asian                       | 7               |
    +----------------------+---------------------------------------+-----------------+
    | Black                | Caribbean                             | 14              |
    |                      +---------------------------------------+-----------------+
    |                      | African                               | 15              |
    |                      +---------------------------------------+-----------------+
    |                      | Other black background                | 16              |
    |                      +---------------------------------------+-----------------+
    |                      | White and Black Caribbean             | 5               |
    |                      +---------------------------------------+-----------------+
    |                      | White and Black African               | 6               |
    +----------------------+---------------------------------------+-----------------+
    | Arab                                                         | 17              |
    +----------------------+---------------------------------------+-----------------+
    | Other                | Any other ethnic group                | 97              |
    |                      +---------------------------------------+-----------------+
    |                      | Any other mixed background            | 8               |
    +----------------------+---------------------------------------+-----------------+
    | invalid as above                                             | < 0             |
    +----------------------+---------------------------------------+-----------------+

    ``distance`` is coded as follows for `respondent-nominee pairs
    <https://www.understandingsociety.ac.uk/documentation/mainstage/dataset-documentation/wave/3/
    datafile/c_indresp/variable/c_netlv_1>`__. The reference distribution for controls needs to be
    obtained separately, e.g. using Monte Carlo simulation.

    +-------------------------------------------------+------+
    | Value                                           | Code |
    +=================================================+======+
    | less than one mile                              | 1    |
    +-------------------------------------------------+------+
    | less than five miles                            | 2    |
    +-------------------------------------------------+------+
    | between five and fifty miles                    | 3    |
    +-------------------------------------------------+------+
    | over fifty miles but still within the uk        | 4    |
    +-------------------------------------------------+------+
    | [excluded] this friend lives in another country | 5    |
    +-------------------------------------------------+------+
    | invalid as above                                | < 0  |
    +-------------------------------------------------+------+

    ``relative`` captures whether `respondent-nominee pairs are related
    <https://www.understandingsociety.ac.uk/documentation/mainstage/dataset-documentation/wave/3/
    datafile/c_indresp/variable/c_netwr_1>`__ to one another; relatives are excluded.

    +--------------------------------------------------+------+
    | Value                                            | Code |
    +==================================================+======+
    | yes                                              | 1    |
    +--------------------------------------------------+------+
    | no                                               | 2    |
    +--------------------------------------------------+------+
    | invalid as above                                 | < 0  |
    +--------------------------------------------------+------+
    """
    raw = pd.read_stata(filename, convert_categoricals=False)

    ego_attrs = {
        'occupation': '%s_jbstat',
        'age': '%s_dvage',
        'sex': '%s_sex',
        'ethnicity': '%s_racel_dv',
        'weight': '%s_indscub_xw',
    }

    alter_attrs = {
        'sex': '%s_netsx_%d',
        'age': '%s_netag_%d',
        'distance': '%s_netlv_%d',
        'occupation': '%s_netjb_%d',
        'ethnicity': '%s_netet_%d',
        'relative': '%s_netwr_%d',
    }

    common_coding = {
        'sex': {
            1: 'male',
            2: 'female',
        }
    }

    skipped_egos = []
    skipped_alters = []
    z = []
    egos = []
    pairs = []

    for _, row in raw.iterrows():
        ego = {key: row[value % code] for key, value in ego_attrs.items()}
        # Skip egos with zero weight
        if ego['weight'] <= 0 or np.isnan(ego['weight']):
            continue
        # Skip egos with missing values or "doing something else"
        if any(value < 0 for value in ego.values()) or ego['occupation'] == 97:
            skipped_egos.append(ego)
            continue
        ego = _recode_ethnicity(ego)
        ego = util.recode(ego, **util.expand_mappings(
            occupation={
                (1, 2, 5, 10): 'employed',
                (3, 8): 'unemployed',
                (7, 9, 11): 'education',
                6: 'family care',
                4: 'retired',
            }, **common_coding,
        ))

        ego_idx = len(z)
        egos.append(ego_idx)
        z.append(ego)

        # Iterate over the alters
        for i in range(3):
            alter = {key: row[value % (code, i + 1)] for key, value in alter_attrs.items()}
            # Drop relatives
            if alter.pop('relative') == 1:
                continue

            # Skip alters with missing values or who live outside the UK
            if any(value < 0 for value in alter.values()) or alter['distance'] == 5:
                skipped_alters.append(alter)
                continue
            alter = _recode_ethnicity(alter)
            alter = util.recode(alter, **util.expand_mappings(
                occupation={
                    (1, 2): 'employed',
                    3: 'unemployed',
                    4: 'education',
                    5: 'family care',
                    6: 'retired',
                }, **common_coding,
            ))

            alter_idx = len(z)
            z.append(alter)
            pairs.append((alter_idx, ego_idx))

    n = {
        'c': 63.5e6,  # https://www.google.com/search?q=2012+uk+population
        'f': 64.85e6,  # https://www.google.com/search?q=2015+uk+population
    }

    sampled_distances = np.squeeze(np.loadtxt(distance_filename)) / 1609

    return {
        'egos': egos,
        'pairs': pairs,
        'z': util.to_records(z),
        'feature_map': ft.partial(understanding_society_feature_map,
                                  sampled_distances=sampled_distances),
        'weights': 'weight',
        'n': n[code],
    }
