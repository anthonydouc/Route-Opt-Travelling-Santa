# -*- coding: utf-8 -*-
from routeopt import find_tsp_route, save_route

if __name__ == '__main__':

    # number of clusters to divide cities into
    ncluster = 500
    
    # number of heuristic search steps to apply for the coarse path between
    # city clusters
    niter_bcl = 1000000
    
    # number of heauristic search steps to apply for the path between
    # cities within each cluster
    niter_wcl = 1000000
    
    route, d = find_tsp_route(ncluster, niter_bcl, niter_wcl)

    print(f'Found path with total length: {d}')

    save_route(route, 'route')
