# -*- coding: utf-8 -*-
'''
Module for reading and processing city data.
'''

import numpy as np
import pandas as pd

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


def get_node_positions(data):
    '''Returns the location (x,y) of each city in the data.'''
    return data['X'].values, data['Y'].values


def get_node_primality(data):
    '''Returns the primality of each city in the data. '''
    return data['Prime_city'].values


def read_data() -> pd.DataFrame:
    '''
    Reads raw city location data provided by Kaggle.
    https://www.kaggle.com/competitions/traveling-santa-2018-prime-paths/data.
    '''

    data = pd.read_csv('data/raw/cities.csv')

    return data


def assign_cluster(cities: pd.DataFrame, ncluster: int=100) -> (pd.DataFrame, pd.DataFrame):
    '''
    Assigns cities into groups by KMEANS clustering on the position of each city.
    '''

    X, Y = get_node_positions(cities)

    clustering = KMeans(n_clusters=ncluster, random_state=0).fit(np.array([X, Y]).T)

    cities['cluster'] = clustering.labels_

    centers = pd.DataFrame(clustering.cluster_centers_, columns=['X', 'Y'])

    # set the cluster id containing city ID to 0.

    cluster_c0 = cities.loc[cities['CityId'] == 0, 'cluster'].values[0]

    m_cl0 = cities['cluster'] == 0

    m_c0 = cities['cluster'] == cluster_c0

    cities.loc[m_cl0, 'cluster'] = cluster_c0

    cities.loc[m_c0, 'cluster'] = 0

    # swap the cluster centers

    centers_cl0 = centers.loc[0, ['X', 'Y']]

    centers_c0 = centers.loc[cluster_c0, ['X', 'Y']]

    centers.loc[0, ['X', 'Y']] = centers_c0

    centers.loc[cluster_c0, ['X', 'Y']] = centers_cl0

    return cities, centers


def get_data(used_saved: bool=True, ncluster:int=100):
    '''
    Returns processed city data. If available, data will be read from file.
    '''

    cities_filename = f'data/processed/cities_{ncluster}.csv'

    centers_filename = f'data/processed/centers_{ncluster}.csv'

    if used_saved and (Path(cities_filename).exists() and Path(centers_filename).exists()):
        cities = pd.read_csv(cities_filename)

        centers = pd.read_csv(centers_filename)

    else:
        cities = read_data()

        primes = get_primes_seive(len(cities))

        cities['Prime_city'] = cities.index.isin(primes)

        cities, centers = assign_cluster(cities, ncluster)

        Path("data/processed").mkdir(parents=True, exist_ok=True)

        cities.to_csv(cities_filename)

        centers.to_csv(centers_filename)

    return cities, centers


def save_route(path: list, filename: str='route'):

    Path("routes").mkdir(parents=True, exist_ok=True)

    save_data = pd.DataFrame(path, columns=['Path'])

    save_data.to_csv(f"routes/{filename}.csv", index=False)


def load_route(filename:str='route'):

    route = pd.read_csv(f"routes/{filename}.csv")

    return list(route['Path'])
