import logging
import numpy as np
import pandas as pd
from . import util


LOGGER = logging.getLogger(__name__)


def general_social_survey_feature_map(x, y):
    """
    Evaluate features for the General Social Survey.
    """
    return util.to_records({
        'bias': np.ones(x.shape[0]),
        'sex': x['sex'] != y['sex'],
        'age': np.abs(x['age'] - y['age']),
        'educ': np.abs(x['educ'] - y['educ']),
        'ethnicity': x['ethnicity'] != y['ethnicity'],
        'relig': x['relig'] != y['relig'],
    })


def load_general_social_survey(filename):
    """
    Load the dataset from the General Social Survey 2004.

    Notes
    -----
    This section discusses coding for both respondents and nominees and details the steps taken to
    harmonise codings.

    ``sex`` is coded as follows for
    `respondents <https://gssdataexplorer.norc.org/variables/81/vshow>`__ and
    `nominees <https://gssdataexplorer.norc.org/variables/865/vshow>`__.

    +---------------+--------------+------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value | Respondent Code |
    +===============+==============+==================+=================+
    | male          | 1            | male             | 1               |
    +---------------+--------------+------------------+-----------------+
    | female        | 2            | female           | 2               |
    +---------------+--------------+------------------+-----------------+
    | no answer     | 9            | ---                                |
    +---------------+--------------+------------------+-----------------+

    ``ethnicity`` is coded as follows for
    `respondents <https://gssdataexplorer.norc.org/variables/5277/vshow>`__ and
    `nominees <https://gssdataexplorer.norc.org/variables/870/vshow>`__.

    +---------------+--------------+----------------------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value                 | Respondent Code |
    +===============+==============+==================================+=================+
    | Asian         | 1            | Asian indian                     | 4               |
    |               |              +----------------------------------+-----------------+
    |               |              | Chinese                          | 5               |
    |               |              +----------------------------------+-----------------+
    |               |              | Filipino                         | 6               |
    |               |              +----------------------------------+-----------------+
    |               |              | Japanese                         | 7               |
    |               |              +----------------------------------+-----------------+
    |               |              | Korean                           | 8               |
    |               |              +----------------------------------+-----------------+
    |               |              | Vietnamese                       | 9               |
    |               |              +----------------------------------+-----------------+
    |               |              | Other Asian                      | 10              |
    +---------------+--------------+----------------------------------+-----------------+
    | Black         | 2            | Black                            | 2               |
    +---------------+--------------+----------------------------------+-----------------+
    | Hispanic      | 3            | Hispanic                         | 16              |
    +---------------+--------------+----------------------------------+-----------------+
    | White         | 4            | White                            | 1               |
    +---------------+--------------+----------------------------------+-----------------+
    | Other         | 5            | American Indian or Alaska Native | 3               |
    |               |              +----------------------------------+-----------------+
    |               |              | Native Hawaiian                  | 11              |
    |               |              +----------------------------------+-----------------+
    |               |              | Guamanian or Chamorro            | 12              |
    |               |              +----------------------------------+-----------------+
    |               |              | Samoan                           | 13              |
    |               |              +----------------------------------+-----------------+
    |               |              | Other Pacific Islander           | 14              |
    |               |              +----------------------------------+-----------------+
    |               |              | Some other race                  | 15              |
    +---------------+--------------+----------------------------------+-----------------+
    | don't know    | 8            | don't know                       | 98              |
    +---------------+--------------+----------------------------------+-----------------+
    | no answer     | 9            | no answer                        | 99              |
    +---------------+--------------+----------------------------------+-----------------+

    ``education`` is coded in multiple fields for respondents (
    `number of years in education <https://gssdataexplorer.norc.org/variables/55/vshow>`__ and the
    `highest degree achieved <https://gssdataexplorer.norc.org/variables/59/vshow>`__). Educational
    coding for `nominees <https://gssdataexplorer.norc.org/variables/940/vshow>`__ is categorical.

    +--------------------------+--------------+----------------------------------+-----------------+
    | Nominee Value            | Nominee Code | Respondent Value                 | Respondent Code |
    +==========================+==============+==================================+=================+
    | 1--6 years               | 0            | No HS diploma, 1--6 years        | 0               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | 7--9 years               | 1            | No HS diploma, 7--9 years        | 0               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | 10--12 years             | 2            | No HS diploma, 10--12 years      | 0               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | High school graduate     | 3            | HS diploma, 12 years             | 0               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | Some college             | 4            | HS diploma, no Bachelor's        | 1               |
    |                          |              | degree, > 12 years               |                 |
    +--------------------------+--------------+----------------------------------+-----------------+
    | Associate degree         | 5            | Junior college                   | 2               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | Bachelor's degree        | 6            | Bachelor's degree                | 3               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | Graduate or professional | 7            | Graduate degree                  | 4               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | don't know               | 8            | don't know                       | 8               |
    +--------------------------+--------------+----------------------------------+-----------------+
    | no answer                | 9            | no answer                        | 9               |
    +--------------------------+--------------+----------------------------------+-----------------+

    ``age`` is coded as a numerical value up to 89 for
    `respondents <https://gssdataexplorer.norc.org/variables/53/vshow>`__ and up to 91 for
    `nominees <https://gssdataexplorer.norc.org/variables/945/vshow>`__.

    +---------------+--------------+------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value | Respondent Code |
    +===============+==============+==================+=================+
    | numeric up to 88             | numeric up to 91                   |
    +---------------+--------------+------------------+-----------------+
    | 89 or older   | 89           | ---                                |
    +---------------+--------------+------------------+-----------------+
    | don't know    | 98           | don't know       | 98              |
    +---------------+--------------+------------------+-----------------+
    | no answer     | 99           | no answer        | 99              |
    +---------------+--------------+------------------+-----------------+

    ``religion`` is coded as follows for
    `respondents <https://gssdataexplorer.norc.org/variables/287/vshow>`__ and
    `nominees <https://gssdataexplorer.norc.org/variables/950/vshow>`__.

    +---------------+--------------+-------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value  | Respondent Code |
    +===============+==============+===================+=================+
    | Protestant    | 1            | Protestant        | 1               |
    +---------------+--------------+-------------------+-----------------+
    | Catholic      | 2            | Catholic          | 2               |
    +---------------+--------------+-------------------+-----------------+
    | Jewish        | 3            | Jewish            | 3               |
    +---------------+--------------+-------------------+-----------------+
    | None          | 4            | None              | 4               |
    +---------------+--------------+-------------------+-----------------+
    | Other         | 5            | Other             | 5               |
    |               |              +-------------------+-----------------+
    |               |              | Buddhism          | 6               |
    |               |              +-------------------+-----------------+
    |               |              | Hinduism          | 7               |
    |               |              +-------------------+-----------------+
    |               |              | Other Eastern     | 8               |
    |               |              +-------------------+-----------------+
    |               |              | Islam             | 9               |
    |               |              +-------------------+-----------------+
    |               |              | Orthodox Christ.  | 10              |
    |               |              +-------------------+-----------------+
    |               |              | Christian         | 11              |
    |               |              +-------------------+-----------------+
    |               |              | Native American   | 12              |
    |               |              +-------------------+-----------------+
    |               |              | Nondenominational | 13              |
    +---------------+--------------+-------------------+-----------------+
    | don't know    | 8            | don't know        | 98              |
    +---------------+--------------+-------------------+-----------------+
    | no answer     | 9            | no answer         | 99              |
    +---------------+--------------+-------------------+-----------------+
    """
    raw = pd.read_stata(filename, convert_categoricals=False)

    # Attributes to fetch
    ego_attrs = [
        'sex',
        'age',
        'racecen1',
        'educ', 'degree',
        'relig',
        'wtss',  # sampling weights
    ]
    alter_attrs = {
        'sex': 'sex%d',
        'age': 'age%d',
        'ethnicity': 'race%d',
        'educ': 'educ%d',
        'relig': 'relig%d',
    }
    # Generic values to recode to missing
    nulls = {
        'sex': [8, 9],
        'age': [98, 99],
        'degree': [8, 9],
        'ethnicity': [8, 9],
        'racecen1': [98, 99],
    }

    # Attributes of the individuals
    z = []
    egos = []
    alters = []
    pairs = []
    egos_skipped = []
    alters_skipped = []

    for _, row in raw.iterrows():
        # Get the attributes of the ego
        ego = {attr: row[attr] for attr in ego_attrs}
        ego = util.fill_values(ego, **nulls, educ=[98, 99], relig=[98, 99])

        if any(map(pd.isnull, ego.values())) or not ego['wtss']:
            egos_skipped.append(ego)
            continue

        # Remap values to ensure consistency with the alters as described in the docstring
        ego = util.recode(ego, **util.expand_mappings(**{
            'racecen1': {
                (4, 5, 6, 7, 8, 9, 10): 'asian',
                2: 'black',
                16: 'hispanic',
                1: 'white',
                (3, 11, 12, 13, 14, 15): 'other',
            },
            'relig': {
                # Protestant, Catholic, Jewish, and None are consistently coded
                1: 'protestant',
                2: 'catholic',
                3: 'jewish',
                4: 'none',
                (5, 6, 7, 8, 9, 10, 11, 12, 13): 'other',  # Other
            },
            'sex': {
                1: 'male',
                2: 'female',
            }
        }))
        ego['ethnicity'] = ego.pop('racecen1')

        # The education mapping is a bit fiddly
        years = ego.pop('educ')
        degree = ego.pop('degree')

        if degree == 0:
            if years <= 6:
                educ = 0
            elif years <= 9:
                educ = 1
            elif years <= 12:
                educ = 2
        elif degree == 1:
            if years <= 12:
                educ = 3  # High school degree
            else:
                educ = 4  # Some college
        ego['educ'] = {
            2: 5,  # Associate
            3: 6,  # Bachelor
            4: 7,  # Graduate
        }.get(degree, educ)

        # Add the ego
        ego_idx = len(z)
        z.append(ego)
        egos.append(ego_idx)

        # Get the number of alters and iterate (up to five)
        num = row['numgiven']
        num = 0 if pd.isnull(num) else int(num)
        for i in range(1, 1 + min(num, 5)):
            # Get the attributes of the alter
            alter = {key: row[value % i] for key, value in alter_attrs.items()}
            alter = util.fill_values(alter, **nulls, educ=[8, 9, -1], relig=[8, 9])
            # Skip straight away if all values are missing
            if all(map(pd.isnull, alter.values())):
                continue
            # Record alters with missing information if some data are present
            if any(map(pd.isnull, alter.values())):
                alters_skipped.append(alter)
                continue

            # Check the relationship type and constrain to friend
            keys = ['spouse', 'parent', 'sibling', 'child', 'othfam', 'cowork', 'memgrp',
                    'neighbr', 'friend', 'advisor', 'other']
            types = {key: row['%s%d' % (key, i)] == 1 for key in keys}

            family = ['spouse', 'parent', 'sibling', 'child', 'othfam']
            if any(types[key] for key in family) or not types['friend']:
                continue

            alter = util.recode(alter, relig={
                1: 'protestant',
                2: 'catholic',
                3: 'jewish',
                4: 'none',
                5: 'other',
            }, ethnicity={
                1: 'asian',
                2: 'black',
                3: 'hispanic',
                4: 'white',
                5: 'other',
            }, sex={
                1: 'male',
                2: 'female',
            })

            alter_idx = len(z)
            z.append(alter)
            pairs.append((alter_idx, ego_idx))
            alters.append(alter_idx)

    # TODO: recode values into something interpretable, like "male" instead of `1`

    # TODO transfer one level up
    LOGGER.info('skipped %d egos', len(egos_skipped))
    LOGGER.info('skipped %d alters', len(alters_skipped))

    return {
        'z': util.to_records(z),
        'egos': egos,
        'pairs': pairs,
        'weights': 'wtss',
        'n': 292.8e6,  # https://www.google.com/search?q=2004+us+population
        'feature_map': general_social_survey_feature_map,
    }
