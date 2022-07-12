# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from operators import local_search

from scipy.spatial import KDTree

from data import calc_euclidean_dist


# e.g step 9 could be a prime, but we would want this to be non prime,
# and make step 10 a prime if swapping is better (10% better + random change due to different path).


def get_edges_from_path(path):
    return [(path[i], path[i+1]) for i in range(0, len(path) - 1)]


def nearest_neighbour(nodes, X, Y, start: int=None, end: int=None):

    d = calc_euclidean_dist(X, Y)

    if start is None:
        start = nodes[0]
    if end is None:
        end = nodes[0]

    n = len(nodes)
    
    if start == end:
        ns = n 
    else:
        ns = n -1

    path = np.array([start], dtype=int)

    steps = range(0, ns)

    for step in steps:

        node = path[-1]

        if step == steps[-1]:
            path = np.append(path, end)
        else:

            valid_idx = nodes[~np.in1d(nodes, np.append(path, end))]
            
            dv = d[node][valid_idx]
                
            if len(dv) > 0:
                
                v = valid_idx[np.argmin(dv)]

                path = np.append(path, v)

    edges = get_edges_from_path(path)

    return path, edges


def find_opt_path(nodes: list, X:np.array, Y: np.array, start: int,
                  end: int, niter: int=1000) ->  (list, list):
    '''
    Finds the shortest possible path based on local optimisation of
    an initial starting path.
    '''

    path, edges = nearest_neighbour(nodes, X, Y, start, end)
    
    path = local_search(path, X, Y, niter)
    
    edges = get_edges_from_path(path)
    
    return path, edges



def run_course_opt(cities: pd.DataFrame, centers: pd.DataFrame,
                   niter: int=1000):

    nodes = centers.index.values
    
    X, Y = centers['X'], centers['Y']
    
    path, edges = find_opt_path(nodes, X, Y, 0, 0, niter)

    cluster_endpoints = get_endpoints(cities, edges)
    
    return path, edges, cluster_endpoints  


def get_endpoints(cities: pd.DataFrame, cluster_edges: pd.DataFrame):
    ''' Determines nodes connecting each cluster based on the shortest edge
        that can be constructed between each cluster.
        Shortest edge is found by a nearest neighbour search between
        all nodes in cluster 1 and cluster 2, using a KDTree.
    '''

    path_nodes = []
    
    visited = []

    for c1, c2 in cluster_edges:
        
        # start and end nodes cannot be the same within each cluster
        m_visited = cities['CityId'].isin(visited)
        
        nodes_c1 = cities[(cities['cluster'] == c1) & (~m_visited)]

        nodes_c2 = cities[(cities['cluster'] == c2) & (~m_visited)]

        X1 = nodes_c1[['X', 'Y']].values

        X2 = nodes_c2[['X', 'Y']].values

        tree = KDTree(X2)

        mindist, minid = tree.query(X1)

        id_en1 = np.argmin(mindist) # index of node in c1 that gives min distance

        id_en2 = minid[id_en1] # index of node in c2 that gives min distance

        en1 = nodes_c1.iloc[id_en1]['CityId'] # city id in c1

        en2 = nodes_c2.iloc[id_en2]['CityId'] # city id in c2

        path_nodes.append((en1, en2))
        
        visited += [en1, en2]

    path_nodes[-1] = (path_nodes[-1][0], 0)

    return path_nodes

  


def opt_cluster(cities, cluster_endpoints, cluster_edges, cluster: int):

    nodes_c = cities[cities['cluster'] == cluster]

    start, end = cluster_endpoints[cluster]

    c_lk = pd.DataFrame(cluster_edges, columns=['From', 'To'])

    cl_start = c_lk.loc[c_lk['To'] == cluster].index[0]

    cl_end = c_lk.loc[c_lk['From'] == cluster].index[0]

    n_start = cluster_endpoints[cl_start][1]

    n_end = cluster_endpoints[cl_end][0]

    X, Y = nodes_c['X'].values, nodes_c['Y'].values

    nodes = nodes_c['CityId'].values

    node_ids = np.array((range(0, len(nodes))))

    n_start_id = np.where(nodes==n_start)[0][0]
    
    n_end_id = np.where(nodes==n_end)[0][0]
    
    path, edges = find_opt_path(node_ids, X, Y, n_start_id, n_end_id, 1000)
    
    path = nodes[path]

    return path


def run_cluster_opt(cities, cluster_endpoints, cluster_edges, clusters):
    print('running!')
    path = []
    for cluster in clusters:
        print(cluster)
        c_path = opt_cluster(cities, cluster_endpoints, cluster_edges, cluster)
        path += list(c_path)
    return path

