# -*- coding: utf-8 -*-

from data import (get_data, calc_euclidean_dist, create_submission,
                  get_node_positions, save_tour, load_tour)

from lpalgorithms import get_path_mtz

from algorithms import nearest_neighbour, get_endpoints, opt_cluster

from operators import local_search 

from multiprocessing import Pool

# if there is time, could run MTZ with a time limit. Use solution if found.
def run_cluster_opt(cities, cluster_endpoints, cluster_edges, clusters):
    print('running!')
    path = []
    for cluster in clusters:
        print(cluster)
        c_path = opt_cluster(cities, cluster_endpoints, cluster_edges, cluster)
        path += list(c_path)
    return path

print('main...')
# partition across 8 threads

if __name__ == '__main__':

    cities, centers = get_data(used_saved=True, ncluster=500)
    
    pX, pY = get_node_positions(cities)
    
    nodes = centers.index.values
    
    X, Y = centers['X'], centers['Y']
    
    cluster_path, cluster_edges = nearest_neighbour(nodes, X, Y, 0, 0)
    
    cluster_path = local_search(cluster_path, X, Y, 10000000)
    print('blah!')
    cluster_endpoints = get_endpoints(cities, cluster_edges)
    print('gah!')
    
    path = []

    all_clusters = list(cluster_path[:-1])

    ncl = len(all_clusters)

    ncore = 8

    ncl_p = ncl // ncore

    cluster_jobs = [all_clusters[i * ncl_p: (i+1) * ncl_p]
                    for i in range(0, ncore)]

    if ncl % ncore != 0:
        cluster_jobs[-1] += all_clusters[ncore * ncl_p: len(all_clusters)]
        
   

    # can we parallelise this
    with Pool(ncore) as p:
        args = [[cities, cluster_endpoints, cluster_edges, clusters] for clusters in cluster_jobs]
        print('args')
        out = p.starmap(run_cluster_opt, args)
        
    path = [j for sub in out for j in sub]

    path = path + [0]
    
    create_submission(path, 'test')

    from bokeh.plotting import figure
    from bokeh.io import show
    
    fig = figure(output_backend='webgl')
    fig.scatter(x=pX, y=pY, size=0.2, color='red')
    
    fig.square(x=centers['X'].values, y=centers['Y'].values, size=2)
    
    Xl, Yl = centers.loc[cluster_path, 'X'].values, centers.loc[cluster_path, 'Y'].values
    
    fig.line(x=Xl, y=Yl, line_width=4)
    
    #fig.line(x=cities.loc[path,'X'].values, y=cities.loc[path,'Y'].values, line_width=3,
    #          color='green')
    
    show(fig)


