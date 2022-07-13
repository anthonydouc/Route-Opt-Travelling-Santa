# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from data import calc_euclidean_dist
from multiprocessing import Pool, cpu_count

from data import get_data, calc_euclidean_dist, create_submission, save_tour

from route_finding import find_opt_path


def calc_path_distance(edges, pX, pY):

    distance = 0

    for i, (n1, n2) in enumerate(edges):
        x = pX[n2] - pX[n1]
        y = pY[n2] - pY[n1]
        distance += math.sqrt(x * x +  y * y)
        
    return distance


def calc_path_distance_prime(path, pX, pY, isprime):

    distance = []

    for i, n in enumerate(path[:-1]):
        x = pX[path[i+1]] - pX[n]
        y = pY[path[i+1]] - pY[n]
        distance.append(math.sqrt(x * x +  y * y))
        
    # # check every 10th edge. If source node is prime, no penalty.
    penalty = (~isprime[path[10::10]]) * 0.1
        
    distance[10::10] *= (1 + penalty)

    return sum(distance)


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


def opt_cluster(cities, cluster_endpoints, cluster_edges, cluster: int, niter: int):

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

    path, edges = find_opt_path(node_ids, X, Y, n_start_id, n_end_id, niter)

    path = nodes[path]

    return path


def run_cluster_opt(cities, cluster_endpoints, cluster_edges, clusters, niter):
    print('running!')
    path = []
    for cluster in clusters:
        print(cluster)
        c_path = opt_cluster(cities, cluster_endpoints, cluster_edges, cluster, niter)
        path += list(c_path)
    return path


def get_mp_clusters(ncores: int, clusters: list) -> list[list]:
    '''
    Returns a list of clusters to be run in parallel.
    '''

    ncl = len(clusters)

    ncore = 8

    ncl_p = ncl // ncore

    cluster_jobs = [clusters[i * ncl_p: (i+1) * ncl_p]
                    for i in range(0, ncore)]

    if ncl % ncore != 0:
        cluster_jobs[-1] += clusters[ncore * ncl_p: len(clusters)]

    return cluster_jobs


def find_tsp_path(ncluster: int=500, niter_bcl: int=500, niter_wcl: int=500):

    cities, centers = get_data(used_saved=True, ncluster=ncluster)

    cluster_path, cluster_edges, cluster_endpoints = run_course_opt(cities, centers, niter=niter_bcl)

    clusters = list(cluster_path[:-1])

    ncores = cpu_count()

    cluster_jobs = get_mp_clusters(ncores, clusters)

    path = []

    with Pool(ncores) as p:

        args = [[cities, cluster_endpoints, cluster_edges, clusters, niter_wcl]
                for clusters in cluster_jobs]

        out = p.starmap(run_cluster_opt, args)

    path = [j for sub in out for j in sub]

    path = path + [0]

    return path

