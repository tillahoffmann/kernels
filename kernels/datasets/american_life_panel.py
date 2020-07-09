import numpy as np
import pandas as pd
from . import dataset


class AmericanLifePanelDataset(dataset.Dataset):
    """
    Load the dataset from the American Life Panel survey.

    Parameters
    ----------
    filename : str
        Filename from which to load data.

    Notes
    -----
    This section discusses coding for both respondents and nominees and details the steps taken to
    harmonise codings.

    ``sex`` is coded as follows for `respondents
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=2&qnid=115>`__ and
    `nominees
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=43>`__.

    .. note::
       Respondents are asked for their gender but for their nominees' sex. We treat them
       interchangeably to unify codings.

    +--------------------------+-------------------------+
    | Respondent/Nominee Value | Respondent/Nominee Code |
    +==========================+=========================+
    | male                     | 1                       |
    +--------------------------+-------------------------+
    | female                   | 2                       |
    +--------------------------+-------------------------+

    ``age`` is coded as a numeric value for `respondents
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=2&qnid=116>`__, but
    age is coded as an ordinal variable for `nominees
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=46>`__. Ages
    for respondents are aggregated to match the ordinal ages of nominees.

    +---------------+--------------+------------------+-----------------+
    | Nominee Value | Nominee Code | Respondent Value | Respondent Code |
    +===============+==============+==================+=================+
    | 0 - 20        | 1            | numeric                            |
    +---------------+--------------+                                    |
    | 21 - 35       | 2            |                                    |
    +---------------+--------------+                                    |
    | 36 - 50       | 3            |                                    |
    +---------------+--------------+                                    |
    | 51 - 65       | 4            |                                    |
    +---------------+--------------+                                    |
    | 66 - 80       | 5            |                                    |
    +---------------+--------------+                                    |
    | 81 +          | 6            |                                    |
    +---------------+--------------+------------------+-----------------+

    ``state`` of residence is coded as a categorical variable for both respondents (missing webpage
    for the field ``statereside``) and `nominees
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=48>`__.

    ``ethnicity`` is coded as follows for `respondents
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=2&qnid=2028>`__ and
    `nominees
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=52>`__. We
    furthermore refine ethnic identity based on whether the `respondent
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=2&qnid=133>`__ or
    `nominee <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=50>`__
    identified as Hispanic.

    +--------------------------+-----------------------------------------------------+
    | Respondent/Nominee Value | Respondent/Nominee Code                             |
    +==========================+=====================================================+
    | White                    | 1                                                   |
    +--------------------------+-----------------------------------------------------+
    | Black                    | 2                                                   |
    +--------------------------+-----------------------------------------------------+
    | Native                   | 3                                                   |
    +--------------------------+-----------------------------------------------------+
    | Asian                    | 4                                                   |
    +--------------------------+-----------------------------------------------------+
    | Other                    | 5                                                   |
    +--------------------------+-----------------------------------------------------+
    | Hispanic                 | if identified as Hispanic irrespective of ethnicity |
    +--------------------------+-----------------------------------------------------+

    ``education`` is coded as follows for `respondents <https://alpdata.rand.org/
    index.php?page=data&p=showquestion&syid=86&meid=2&qnid=128>`__ and `nominees
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=54>`__.

    +----------------------------------+--------------+--------------------------+-----------------+
    | Nominee Value                    | Nominee Code | Respondent Value         | Respondent Code |
    +==================================+==============+==========================+=================+
    | < 9th grade                      | 1            | < 1st grade              | 1               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 1st to 4th grade         | 2               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 5th or 6th grade         | 3               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 7th or 8th grade         | 4               |
    +----------------------------------+--------------+--------------------------+-----------------+
    | 9th - 12th grade, no diploma     | 2            | 9th grade                | 5               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 10th grade               | 6               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 11th grade               | 7               |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | 12th grade               | 8               |
    +----------------------------------+--------------+--------------------------+-----------------+
    | High school graduate or GED      | 3            | High school graduate or  | 9               |
    |                                  |              | equivalent               |                 |
    +----------------------------------+--------------+--------------------------+-----------------+
    | Some college, no degree          | 4            | Some college, no degree  | 10              |
    +----------------------------------+--------------+--------------------------+-----------------+
    | Associate Degree                 | 5            | Associate degree         | 11              |
    |                                  |              | (vocational)             |                 |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | Associate degree         | 12              |
    |                                  |              | (academic)               |                 |
    +----------------------------------+--------------+--------------------------+-----------------+
    | Bachelor's degree                | 6            | Bachelor's degree        | 13              |
    +----------------------------------+--------------+--------------------------+-----------------+
    | Master's Degree                  | 7            | Master's degree          | 14              |
    +----------------------------------+--------------+--------------------------+-----------------+
    | Professional degree or doctorate | 8            | Professional degree      | 15              |
    |                                  |              +--------------------------+-----------------+
    |                                  |              | Doctorate                | 16              |
    +----------------------------------+--------------+--------------------------+-----------------+

    ``relationship_type`` is coded as follows for `respondent-nominee pairs
    <https://alpdata.rand.org/index.php?page=data&p=showquestion&syid=86&meid=6&qnid=39>`__, and we
    only retain relationships that are not relatives, i.e. ``code > 3``.

    +------------------------------------------------------------+------+
    | Value                                                      | Code |
    +============================================================+======+
    | Spouse/partner                                             | 1    |
    +------------------------------------------------------------+------+
    | Daughter/son                                               | 2    |
    +------------------------------------------------------------+------+
    | Parent                                                     | 3    |
    +------------------------------------------------------------+------+
    | Work or school mate                                        | 4    |
    +------------------------------------------------------------+------+
    | Neighbor                                                   | 5    |
    +------------------------------------------------------------+------+
    | Online friend                                              | 6    |
    +------------------------------------------------------------+------+
    | Friend from organization (such as church, volunteer group) | 7    |
    +------------------------------------------------------------+------+
    | Professional                                               | 8    |
    +------------------------------------------------------------+------+
    | Other friend                                               | 9    |
    +------------------------------------------------------------+------+
    """
    def __init__(self, filename):
        n = 306.8e6  # from https://www.google.com/search?q=2009+us+population
        super(AmericanLifePanelDataset, self).__init__(n, True)
        self.filename = filename

    def recode(self, x, ego):
        y = {}
        for key, value in x.items():
            if isinstance(value, str) and key != 'identifier':
                value = int(value.split()[0])
            y[key] = value

        y = dataset.recode_values(y, state={
            1:  'ALASKA',
            2:  'ALABAMA',
            3:  'ARIZONA',
            4:  'ARKANSAS',
            5:  'CALIFORNIA',
            6:  'COLORADO',
            7:  'CONNECTICUT',
            8:  'DELAWARE',
            9:  'FLORIDA',
            10: 'GEORGIA',
            11: 'HAWAII',
            12: 'IDAHO',
            13: 'ILLINOIS',
            14: 'INDIANA',
            15: 'IOWA',
            16: 'KANSAS',
            17: 'KENTUCKY',
            18: 'LOUISIANA',
            19: 'MAINE',
            20: 'MARYLAND',
            21: 'MASSACHUSETTS',
            22: 'MICHIGAN',
            23: 'MINNESOTA',
            24: 'MISSISSIPPI',
            25: 'MISSOURI',
            26: 'MONTANA',
            27: 'NEBRASKA',
            28: 'NEVADA',
            29: 'NEW HAMPSHIRE',
            30: 'NEW JERSEY',
            31: 'NEW MEXICO',
            32: 'NEW YORK',
            33: 'NORTH CAROLINA',
            34: 'NORTH DAKOTA',
            35: 'OHIO',
            36: 'OKLAHOMA',
            37: 'OREGON',
            38: 'PENNSYLVANIA',
            39: 'RHODE ISLAND',
            40: 'SOUTH CAROLINA',
            41: 'SOUTH DAKOTA',
            42: 'TENNESSEE',
            43: 'TEXAS',
            44: 'UTAH',
            45: 'VERMONT',
            46: 'VIRGINIA',
            47: 'WASHINGTON',
            48: 'WEST VIRGINIA',
            49: 'WISCONSIN',
            50: 'WYOMING',
            51: 'WASHINGTON D.C.',
            52: 'PUERTO RICO',
        }, sex={
            1: 'male',
            2: 'female',
        }, ethnicity={
            1: 'white',
            2: 'black',
            3: 'native',
            4: 'asian',
            5: 'other',
        }, hispanic={
            1: True,
            2: False,
        },)

        if ego:
            # Map the age to the same categorical variables
            age = y.pop('age')
            if age < 21:
                age = 1
            elif age < 36:
                age = 2
            elif age < 51:
                age = 3
            elif age < 66:
                age = 4
            elif age < 81:
                age = 5
            else:
                age = 6
            y['age'] = age

            y = dataset.recode_values(y, education={
                (1, 2, 3, 4): 1,
                (5, 6, 7, 8): 2,
                9: 3,
                10: 4,
                (11, 12): 5,
                13: 6,
                14: 7,
                (15, 16): 8,
            })
        else:
            y = dataset.recode_values(y, relationship={
                1: 'partner',
                2: 'child',
                3: 'parent',
                4: 'work/school mate',
                5: 'neighbour',
                6: 'online friend',
                7: 'friend from organisation',
                8: 'professional',
                9: 'other',
            })

        # Convert to a year-scale (the offset doesn't matter because we consider L1 features)
        y['age'] = 15 * y['age']
        hispanic = y.pop('hispanic')
        if hispanic is True:  # need to guard for np.nan being truth-y
            y['ethnicity'] = 'hispanic'

        return y

    def feature_map(self, x, y):
        return dataset.to_records({
            'bias': np.ones(x.shape[0]),
            'sex': x['sex'] != y['sex'],
            'age': np.abs(x['age'] - y['age']),
            'ethnicity': x['ethnicity'] != y['ethnicity'],
            'distance': x['state'] != y['state'],
            'education': np.abs(x['education'] - y['education']),
        })

    def load(self):
        raw = pd.read_stata(self.filename)

        ego_attributes = {
            'sex': 'gender',
            'age': 'calcage',
            'state': 'statereside',
            'education': 'highesteducation',
            'ethnicity': 'ethnicity',
            'hispanic': 'hispaniclatino',
            'identifier': 'prim_key',
            'weight': 'weight',
        }
        alter_attributes = {
            'sex': 'sex_%d_',
            'age': 'dm_age_%d_',
            'state': 'state_%d_',
            'education': 'edu_%d_',
            'ethnicity': 'race_%d_',
            'hispanic': 'hisp_%d_',
            'relationship': 'relation_%d_',
        }

        for _, row in raw.iterrows():
            with self.add_ego(self.get_attributes(row, ego_attributes)) as ego_idx:
                if not ego_idx:
                    continue

                for i in range(5):
                    self.add_alter(self.get_attributes(row, alter_attributes, i + 1))

        return super(AmericanLifePanelDataset, self).load()

    def is_invalid(self, x, ego):
        if not ego:
            excluded = ['partner', 'child', 'parent', 'other']
            if pd.isnull(x['relationship']) or x['relationship'] in excluded:
                return 'relationship: %s' % x['relationship']
        return super(AmericanLifePanelDataset, self).is_invalid(x, ego)
