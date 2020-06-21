import numpy as np
import pandas as pd
from . import dataset

ETHNICITY_MAP = {
    'white': (1, 2, 3, 4, 5, 6, 7),
    'asian': (9, 10, 11, 12, 13, 7),
    'black': (14, 15, 16, 5, 6),
    'arab': (17,),
    'other': (8, 97),
}


class UnderstandingSocietyDataset(dataset.Dataset):
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
    def __init__(self, filename, code, distance_filename):
        # cf https://www.understandingsociety.ac.uk/documentation/mainstage/survey-timeline
        population_sizes = {
            'usoc_c': 63.5e6,  # https://www.google.com/search?q=2012+uk+population
            'usoc_f': 64.85e6,  # https://www.google.com/search?q=2015+uk+population
            'usoc_bb': 57.51e6,  # https://www.google.com/search?q=1992+uk+population
            'usoc_bd': 57.79e6,  # https://www.google.com/search?q=1994+uk+population
            'usoc_bf': 58.09e6,  # https://www.google.com/search?q=1996+uk+population
            'usoc_bh': 58.39e6,  # https://www.google.com/search?q=1998+uk+population
            'usoc_bj': 58.79e6,  # https://www.google.com/search?q=2000+uk+population
            'usoc_bl': 59.24e6,  # https://www.google.com/search?q=2002+uk+population
            'usoc_bn': 59.79e6,  # https://www.google.com/search?q=2004+uk+population
            'usoc_bp': 60.62e6,  # https://www.google.com/search?q=2006+uk+population
            'usoc_br': 61.57e6,  # https://www.google.com/search?q=2008+uk+population
        }
        super(UnderstandingSocietyDataset, self).__init__(population_sizes[code], True)
        self.filename = filename
        self.code = code
        _, self.short_code = self.code.split('_')
        self.distance_filename = distance_filename
        self.distance_proba = None
        # Whether we're in the bhps waves or not
        self.is_usoc = len(self.short_code) == 1

    def recode(self, x, ego):
        x = {key: None if value < 0 else value for key, value in x.items()}
        # Recode ethnicities for both alters and egos
        if self.is_usoc:
            ethnicity = x.pop('ethnicity')
            if pd.isnull(ethnicity):
                ethnicities = {key: np.nan for key in ETHNICITY_MAP}
                norm = 1
            else:
                ethnicities = {key: ethnicity in value for key, value in ETHNICITY_MAP.items()}
                norm = sum(ethnicities.values())
                assert norm, "no ethnicity recovered from code %s" % ethnicity
            x.update({'ethnicity_%s' % key: value / norm for key, value in ethnicities.items()})

        x = dataset.recode_values(x, sex={
            1: 'male',
            2: 'female',
        })

        if ego:
            x = dataset.recode_values(x, occupation={
                (1, 2, 5, 10): 'employed',
                (3, 8): 'unemployed',
                (7, 9, 11): 'education',
                6: 'family care',
                4: 'retired',
                97: 'something else',
            })
        else:
            x = dataset.recode_values(x, occupation={
                (1, 2): 'employed',
                3: 'unemployed',
                4: 'education',
                5: 'family care',
                6: 'retired',
            }, relative={
                1: True,
                2: False,
            })

        return x

    def is_invalid(self, x, ego):
        if ego:
            if x['occupation'] == 'something else':
                return 'job: something else'
        else:
            if x['relative']:
                return 'is relative'
            elif x['distance'] == 5:
                return 'outside UK'
        return super(UnderstandingSocietyDataset, self).is_invalid(x, ego)

    def load(self):
        # Load the sampled distances
        sampled_distances = np.squeeze(np.loadtxt(self.distance_filename)) / 1609.34
        proba, _ = np.histogram(sampled_distances, bins=[0, 1, 5, 50, 1e6])
        self.distance_proba = proba / np.sum(proba)

        # Load the main survey dataset
        raw = pd.read_stata(self.filename, convert_categoricals=False)

        ego_attributes = {
            'occupation': '%s_jbstat',
            'age': '%s_age_dv',
            'sex': '%s_sex',
        }

        if self.is_usoc:
            ego_attributes.update({
                'ethnicity': '%s_racel_dv',
                'weight': '%s_indscub_xw',
            })
            alter_attributes = {
                'sex': '%s_netsx_%d',
                'age': '%s_netag_%d',
                'distance': '%s_netlv_%d',
                'occupation': '%s_netjb_%d',
                'relative': '%s_netwr_%d',
                'ethnicity': '%s_netet_%d',
            }
        else:
            ego_attributes.update({
                'weight': '%s_xrwght',
            })
            alter_attributes = {
                'sex': '%s_netsx%d',
                'age': '%s_net%dag',
                'distance': '%s_net%dlv',
                'occupation': '%s_net%djb',
                'relative': '%s_net%dwr',
            }

        for _, row in raw.iterrows():
            with self.add_ego(self.get_attributes(row, ego_attributes, self.short_code)) as ego_idx:
                if not ego_idx:
                    continue
                for i in range(3):
                    alter = self.get_attributes(row, alter_attributes, self.short_code, i + 1)
                    self.add_alter(alter)

        return super(UnderstandingSocietyDataset, self).load()

    def feature_map(self, x, y):
        # Sample distances for controls
        missing = np.isnan(x['distance'])
        sampled = 1 + np.random.choice(len(self.distance_proba), len(x), p=self.distance_proba)
        distance = np.where(missing, sampled, x['distance'])

        features = {
            'bias': np.ones(x.shape[0]),
            'sex': x['sex'] != y['sex'],
            'age': np.abs(x['age'] - y['age']),
            'occupation': x['occupation'] != y['occupation'],
            'distance': distance,
        }

        # Use mixed-membership distance for ethnicities
        if self.is_usoc:
            keys = ['ethnicity_%s' % key for key in ETHNICITY_MAP]
            features['ethnicity'] = 0.5 * sum(abs(x[key] - y[key]) for key in keys)

        return dataset.to_records(features)
