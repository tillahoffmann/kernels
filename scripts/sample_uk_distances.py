import argparse
from atomicwrites import atomic_write
from kernels.util import add_logging_argument, ensure_directory_exists
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from shapefile import Reader as ShapefileReader
from shapely import geometry
import tqdm


def _load_boundaries(filename, index, projections=None, encoding='utf-8'):
    """
    Load boundaries from a shape file and perform a projection if desired.

    Parameters
    ----------
    filename : str
        filename to load data from
    index : int
        index to extract the LSOA identifier from the metadata for each shape
    projections : tuple
        tuple of projections (from, to)

    Returns
    -------
    boundaries : dict
        dictionary of lists of polygons keyed by LSOA code
    """
    boundaries = {}

    # Extract projections
    if projections is not None:
        original_projection, target_projection = projections
        target_projection = pyproj.Proj(init=target_projection)
        original_projection = pyproj.Proj(init=original_projection)
        transformer = pyproj.Transformer.from_proj(original_projection, target_projection)
    else:
        transformer = None

    # Iterate over all records
    shapefile = ShapefileReader(filename, encoding=encoding)
    logging.info("opened shapefile '%s'", filename)
    iterator = shapefile.iterShapeRecords()

    with tqdm.tqdm(total=shapefile.numRecords) as progress:
        while True:
            try:
                sr = next(iterator)
            except IOError:
                # the shapefile module has a bug and raises an error when reading, but the data are
                # fine, and we ignore the error
                break
            except StopIteration:
                break

            # Get the identifier
            code = sr.record[index]

            # Transform the points if transformations are given
            points = sr.shape.points
            if projections:
                points = np.transpose(transformer.transform(*np.transpose(points)))

            # Build a set of polygons and their areas
            polygons = []
            for part in np.array_split(points, sr.shape.parts[1:]):
                polygon = geometry.Polygon(part)
                polygons.append((polygon.area, polygon))

            # Transpose to get a tuple (areas, polygons)
            boundaries[code] = list(zip(*polygons))
            assert len(boundaries[code]) == 2, "expected 2 items for %s but got %d" % \
                (code, len(boundaries[code]))
            progress.update()

    # Check we loaded the correct number of boundaries
    assert len(boundaries) == shapefile.numRecords, "did not load the correct number of records"
    logging.info('loaded %d boundaries from %s', len(boundaries), filename)
    return boundaries


parser = argparse.ArgumentParser()
add_logging_argument(parser)
parser.add_argument('num_samples', help='number of distance samples to obtain', type=int)
parser.add_argument('--filename', help='where to store results')
args = parser.parse_args()


# Load the shapes
boundary_parts = [
    _load_boundaries('data/lsoa_boundaries_2011/LSOA_2011_EW_BGC_V2.shp', 0, encoding='latin-1'),
    _load_boundaries('data/lsoa_boundaries_2011/SOA2011.shp', 0, ('epsg:29902', 'epsg:27700')),
    _load_boundaries('data/lsoa_boundaries_2011/SG_DataZone_Bdry_2011.shp', 1)
]
boundaries = {}
for part in boundary_parts:
    for code, boundary in part.items():
        assert code not in boundaries, 'duplicate code %s' % code
        boundaries[code] = boundary
logging.info('loaded %d shapes' % len(boundaries))

# Get a set of points and produce a scatter plot for sanity checking
points = []
for _, polygons in boundaries.values():
    for polygon in polygons:
        point = polygon.representative_point()
        points.append((point.x, point.y))
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.scatter(*np.transpose(points), marker='.')
fig.savefig(ensure_directory_exists('workspace/lsoas.png'))

# Load population data and get the probability to sample a given LSOA
population = [
    pd.read_excel('data/lsoa_population_2012/mid-2012-lsoa-syoa-unformatted-file.xls',
                  1, usecols=(0, 2)),
    pd.read_excel('data/lsoa_population_2012/2012-sape-t2a-corrected.xlsx',
                  usecols=(0, 2), skiprows=5),
    pd.read_excel('data/lsoa_population_2012/SAPE_SOA_0114.xls', '2012',
                  usecols=(1, 16), skiprows=3),
]
for x in population:
    x.columns = ['code', 'count']
    x.dropna(inplace=True)
population = pd.concat(population)
population['proba'] = population['count'] / population['count'].sum()
assert population['code'].nunique() == len(population), 'encountered non-unique code'
logging.info('loaded %d adminstrative regions' % len(population))

# Actually run the sampling
distances = []
for _ in tqdm.tqdm(range(args.num_samples), desc='sampling'):
    # Sample the administrative regions for the pair
    lsoas = np.random.choice(population.code, 2, p=population.proba)
    points = []
    for lsoa in lsoas:
        # Sample one of the polygons
        areas, polygons = boundaries[lsoa]
        polygon = np.random.choice(polygons, p=areas / np.sum(areas))

        # Sample a point within the polygon
        minx, miny, maxx, maxy = polygon.bounds
        point = None
        for _ in range(500):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            candidate = geometry.Point(x, y)
            if candidate.within(polygon):
                point = candidate
                break
        if point is None:
            raise ValueError('could not sample for %s' % lsoa)
        points.append(point)

    # Compute the distance between points
    a, b = points
    distances.append(a.distance(b))

# Save the results
filename = args.filename
if filename is None:
    filename = f'workspace/uk_distance_samples-{args.num_samples}.txt'
with atomic_write(ensure_directory_exists(filename), overwrite=True) as fp:
    fp.write('\n'.join(map(str, distances)))
