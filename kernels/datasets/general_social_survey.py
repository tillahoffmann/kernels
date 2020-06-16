import numpy as np
import pandas as pd
from . import dataset


class GeneralSocialSurveyDataset(dataset.Dataset):
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
    def __init__(self, filename):
        n = 292.8e6  # https://www.google.com/search?q=2004+us+population
        super(GeneralSocialSurveyDataset, self).__init__(n, True)
        self.filename = filename

    def feature_map(self, x, y):
        return dataset.to_records({
            'bias': np.ones(x.shape[0]),
            'sex': x['sex'] != y['sex'],
            'age': np.abs(x['age'] - y['age']),
            'education': np.abs(x['education'] - y['education']),
            'ethnicity': x['ethnicity'] != y['ethnicity'],
            'religion': x['religion'] != y['religion'],
        })

    def recode(self, x, ego):
        # Common codings
        x = dataset.recode_values(x, sex={
            1: 'male',
            2: 'female',
            (8, 9): None,
        }, religion={
            1: 'protestant',
            2: 'catholic',
            3: 'jewish',
            4: 'none',
            5: 'other',
        })
        # Codings for egos
        if ego:
            x = dataset.recode_values(x, religion={
                (6, 7, 8, 9, 10, 11, 12, 13): 'other',
                (98, 99): None,
            }, ethnicity={
                (4, 5, 6, 7, 8, 9, 10): 'asian',
                2: 'black',
                16: 'hispanic',
                1: 'white',
                (3, 11, 12, 13, 14, 15): 'other',
            }, education={
                (98, 99): None,
            })

            # The education mapping is a bit fiddly
            years = x.pop('education')
            degree = x.pop('degree')
            educ = None
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

            x['education'] = {
                2: 5,  # Associate
                3: 6,  # Bachelor
                4: 7,  # Graduate
            }.get(degree, educ)
        # Codings for alters
        else:
            x = dataset.recode_values(x, ethnicity={
                1: 'asian',
                2: 'black',
                3: 'hispanic',
                4: 'white',
                5: 'other',
            }, education={
                (8, 9, -1): None,
            }, religion={
                (8, 9): None,
            })
            # Parse the relationships
            for key in x:
                if key.startswith('relationship_'):
                    x[key] = x[key] == 1
        return x

    def load(self):
        raw = pd.read_stata(self.filename, convert_categoricals=False)
        ego_attributes = {
            'sex': 'sex',
            'age': 'age',
            'ethnicity': 'racecen1',
            'education': 'educ',
            'degree': 'degree',
            'religion': 'relig',
            'weight': 'wtss'
        }
        alter_attributes = {
            'sex': 'sex%d',
            'age': 'age%d',
            'ethnicity': 'race%d',
            'education': 'educ%d',
            'religion': 'relig%d',
            'relationship_spouse': 'spouse%d',
            'relationship_parent': 'parent%d',
            'relationship_sibling': 'sibling%d',
            'relationship_child': 'child%d',
            'relationship_othfam': 'othfam%d',
            'relationship_cowork': 'cowork%d',
            'relationship_memgrp': 'memgrp%d',
            'relationship_neighbr': 'neighbr%d',
            'relationship_friend': 'friend%d',
            'relationship_advisor': 'advisor%d',
            'relationship_other': 'other%d'
        }

        for _, row in raw.iterrows():
            with self.add_ego(self.get_attributes(row, ego_attributes)) as ego_idx:
                if ego_idx is None:
                    continue
                # Iterate over the alters
                for i in range(5):
                    self.add_alter(self.get_attributes(row, alter_attributes, i + 1))

        return super(GeneralSocialSurveyDataset, self).load()

    def is_invalid(self, x, ego):
        if not ego:
            relatives = ['spouse', 'parent', 'sibling', 'child', 'othfam']
            if any(x['relationship_%s' % key] for key in relatives):
                return 'relative'
            friend = x['relationship_friend']
            if not pd.isnull(friend) and not friend:
                return 'not a friend'
        return super(GeneralSocialSurveyDataset, self).is_invalid(x, ego)
