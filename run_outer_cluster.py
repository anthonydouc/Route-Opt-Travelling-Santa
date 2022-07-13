# -*- coding: utf-8 -*-
from algorithms import find_tsp_path, calc_path_distance_prime, get_edges_from_path

from data import create_submission, get_data, get_node_primality

if __name__ == '__main__':

    path = find_tsp_path(ncluster=435, niter_bcl=100, niter_wcl=100)
        
    cities, centers = get_data(used_saved=True, ncluster=435)
    
    X, Y = cities['X'].values, cities['Y'].values
    
    isprime = get_node_primality(cities)

    d = calc_path_distance_prime(path, X, Y, isprime)
    
    create_submission(path, 'test')
