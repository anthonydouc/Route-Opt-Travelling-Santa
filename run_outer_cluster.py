# -*- coding: utf-8 -*-
from algorithms import find_tsp_path, calc_path_distance_prime#, get_edges_from_path

from route_finding import local_search

from data import create_submission, get_data, get_node_primality

if __name__ == '__main__':
    
    cities, centers = get_data(used_saved=True, ncluster=470)
    
    X, Y = cities['X'].values, cities['Y'].values

    path = find_tsp_path(ncluster=470, niter_bcl=5000000, niter_wcl=1000000)
        
    isprime = get_node_primality(cities)

    d = calc_path_distance_prime(path, X, Y, isprime)
    
    create_submission(path, 'test')
