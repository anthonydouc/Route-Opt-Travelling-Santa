# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import networkx as nx

from lpalgorithms import get_path_mtz
from scipy.spatial import KDTree

from data import calc_euclidean_dist

# Cost calculation func.

# 1. Need a path construction algorithm.

# 2. Then a local optimisation algorithm.

# 3. Then potentially a grand optimnisation alogrithm.

# Explore arbitary rules...  e.g. alpha % of all 10th steps come from a prime.

# Or variant of 2-opts etc... but we look at fixing prime paths.

# e.g step 9 could be a prime, but we would want this to be non prime,
# and make step 10 a prime if swapping is better (10% better + random change due to different path).


# should we use network objects? or just native python objects....
# currently just network for convience graphing


def calc_path_distance(edges, pX, pY):
    # math.sqrt is faster than ** 0.5.
    # x * x can be faster than ** 2.
    distances = []

    for (n1, n2) in edges:
        x = pX[n2] - pX[n1]
        y = pY[n2] - pY[n1]
        distances.append( math.sqrt(x * x +  y * y ) )

    # check every 10th edge. If source node is prime, no penalty.

    return distances


def nearest_index(G, n):
    path = list(range(0, n)) + [0]

    edges = [(path[i-1], path[i]) for i in range(1, len(path))]

    G.add_edges_from(edges)

    return G, path, edges


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

    edges = []

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

    edges = [(path[i], path[i+1]) for i in range(0, ns)]

    return path, edges

# shifts visits of prime cities to the 10th step
# and retains this if the new tour is an improvement
def prime_shift():
    pass



def get_endpoints(cities, cluster_edges):
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

from operators import local_search

def opt_cluster(cities, cluster_endpoints, cluster_edges, cluster: int):

    nodes_c = cities[cities['cluster'] == cluster]

    start, end = cluster_endpoints[cluster]

    c_lk = pd.DataFrame(cluster_edges, columns=['From', 'To'])

    cl_start = c_lk.loc[c_lk['To'] == cluster].index[0]

    cl_end = c_lk.loc[c_lk['From'] == cluster].index[0]

    n_start = cluster_endpoints[cl_start][1]

    n_end = cluster_endpoints[cl_end][0]

    #nodes_rest = nodes_c.loc[~nodes_c['CityId'].isin([n_start, n_end]), 'CityId'].values

    X, Y = nodes_c['X'].values, nodes_c['Y'].values

    nodes = nodes_c['CityId'].values

    node_ids = np.array((range(0, len(nodes))))

    n_start_id = np.where(nodes==n_start)[0][0]
    
    n_end_id = np.where(nodes==n_end)[0][0]
        
    path, edges = nearest_neighbour(node_ids, X, Y, n_start_id, n_end_id)

    #print(path)
    
    path = local_search(path, X, Y, 1000000)

    # path, edges = get_path_mtz(node_ids,
    #                             X,
    #                             Y,
    #                             n_start_id,
    #                             n_end_id)

    path = nodes[path]

    return path

