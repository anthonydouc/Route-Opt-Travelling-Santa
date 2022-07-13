# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from os.path import dirname, join
from pathlib import Path
from sklearn.cluster import KMeans


def get_primes_seive(n: int) -> np.array:
    '''
    Implements the Sieve of Eratosthenes to determine all primes
    from 2 to n.
    '''

    integers = np.linspace(2, n, n-1, dtype=int)

    is_prime = np.ones(n-1, dtype=int)

    p = 2

    for p in range(2, n):

        is_prime[p * 2 - 2::p] = 0

    primes = integers[is_prime==1]

    return primes


def calc_euclidean_dist(x: np.array, y: np.array) -> np.array:
    ''' Calculates the euclidean distance between
        all combination of points contained in the arrays x, y.
    '''
    xs, ys = np.meshgrid(x, y)

    d = ((xs - xs.T) ** 2 + (ys.T - ys) ** 2) ** 0.5

    return d


def read_data() -> pd.DataFrame:

    data = pd.read_csv('data/raw/cities.csv')

    return data


def assign_cluster(data: pd.DataFrame, ncluster: int=100) -> (pd.DataFrame, pd.DataFrame):

    pX, pY = get_node_positions(data)

    clustering = KMeans(n_clusters=ncluster, random_state=0).fit(np.array([pX, pY]).T)

    cluster = clustering.labels_

    center = clustering.cluster_centers_

    data['cluster'] = cluster

    centers = pd.DataFrame(center, columns=['X', 'Y'])

 #   swap the cluster id containing city ID with 0

    z_cluster = data.loc[data['CityId'] == 0, 'cluster'].values[0]

    m_zero = data['cluster'] == 0

    m_z = data['cluster'] == z_cluster

    data.loc[m_zero, 'cluster'] = z_cluster

    data.loc[m_z, 'cluster'] = 0

    # swap the cluster centers

    centers_0 = centers.loc[0, ['X', 'Y']]

    centers_z = centers.loc[z_cluster, ['X', 'Y']]

    centers.loc[0, ['X', 'Y']] = centers_z

    centers.loc[z_cluster, ['X', 'Y']] = centers_0

    return data, centers


def get_data(used_saved: bool=True, ncluster:int=100):

    cities_filename = f'data/processed/cities_{ncluster}.csv'

    centers_filename = f'data/processed/centers_{ncluster}.csv'

    if used_saved and (Path(cities_filename).exists() and Path(centers_filename).exists()):
        cities = pd.read_csv(cities_filename)

        centers = pd.read_csv(centers_filename)

    else:
        cities = read_data()

        cities = cities

        primes = get_primes_seive(len(cities))

        cities['Prime_city'] = cities.index.isin(primes)

        cities, centers = assign_cluster(cities, ncluster)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        cities.to_csv(cities_filename)

        centers.to_csv(centers_filename)

    return cities, centers


def get_node_positions(data):
    return data['X'].values, data['Y'].values


def get_node_primality(data):
    return data['Prime_city'].values


def create_submission(path: list, filename: str='submission'):

    Path("submissions").mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(path, columns=['Path'])

    submission.to_csv(join(dirname(__file__), f"submissions/{filename}.csv"), index=False)


def load_submission(filename:str='submission'):

    paths = pd.read_csv(join(dirname(__file__), f"submissions/{filename}.csv"))

    return list(paths['Path'])


def save_tour(path: list, edges: list, filename: str='best_tour'):

    Path("tours").mkdir(parents=True, exist_ok=True)

    edges = pd.DataFrame(edges, columns=['From', 'To'])

    edges.to_csv(join(dirname(__file__), f"tours/{filename}_edges.csv"), index=False)

    paths = pd.DataFrame(path, columns=['Path'])

    paths.to_csv(join(dirname(__file__), f"tours/{filename}_path.csv"), index=False)


def load_tour(filename:str='best_tour'):

    paths = pd.read_csv(join(dirname(__file__), f"tours/{filename}_path.csv"))

    edges = pd.read_csv(join(dirname(__file__), f"tours/{filename}_edges.csv"))

    return list(paths['Path']), edges[['From', 'To']].values
