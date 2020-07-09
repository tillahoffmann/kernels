import numpy as np
import pandas as pd
import pyproj
from . import dataset


class ResearchNowDataset(dataset.Dataset):
    # Class attribute because it can't be pickled
    transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:27700')

    def __init__(self, filename):
        # https://www.google.com/search?q=2017+uk+population
        super(ResearchNowDataset, self).__init__(65.84e6, False)
        self.filename = filename

    def recode(self, x, ego):
        x = dataset.recode_values(x, religion={
            # People don't seem to care/understand the difference between agnostic and atheist
            ('agnostic', 'atheist'): 'not_religious',
        }, occupation={
            # Recode for consistency with the BHPS
            ('disabled', 'unemployed'): 'not_employed'
        })

        # Convert latitude and longitude into easting and northing

        x['easting'], x['northing'] = self.transformer.transform(x['postcode_latitude'],
                                                                 x['postcode_longitude'])
        return x

    def load(self):
        attributes = {
            'sex': 'gender',
            'age': 'age',
            'religion': 'religion',
            'occupation': 'employment_status',
            'education': 'education',
            'income': 'household_income',
            'postcode_latitude': 'postcode_latitude',
            'postcode_longitude': 'postcode_longitude',
            'ethnicity': 'ethnicity',
        }
        raw = pd.read_csv(self.filename)
        for _, row in raw.iterrows():
            # Skip people who have been referred until later
            if not pd.isnull(row.token):
                continue

            with self.add_ego(self.get_attributes(row, attributes)) as ego_idx:
                if not ego_idx:
                    continue
                alters = raw[raw.token == row.uid]
                for _, alter_row in alters.iterrows():
                    self.add_alter(self.get_attributes(alter_row, attributes))

        return super(ResearchNowDataset, self).load()

    def feature_map(self, x, y):
        # Compute the distance in metres
        dist = np.sqrt(np.square(x['easting'] - y['easting']) +
                       np.square(x['northing'] - y['northing']))
        # Bin in the same fashion as the BHPS
        dist = np.argmax(dist[..., None] < np.asarray([1, 5, 50, 1e6]) * 1609, axis=-1) + 1
        return dataset.to_records({
            'bias': np.ones(x.shape[0]),
            'sex': x['sex'] != y['sex'],
            'age': np.abs(x['age'] - y['age']),
            'occupation': x['occupation'] != y['occupation'],
            'religion': x['religion'] != y['religion'],
            'education': np.abs(x['education'] - y['education']),
            'distance': dist,
            'income': np.abs(x['income'] - y['income']),
            'ethnicity': x['ethnicity'] != y['ethnicity'],
        })
