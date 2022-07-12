# -*- coding: utf-8 -*-
from multiprocessing import Pool

from data import get_data, calc_euclidean_dist, create_submission, save_tour

from algorithms import run_course_opt, run_cluster_opt

if __name__ == '__main__':

    cities, centers = get_data(used_saved=True, ncluster=500)
    
    cluster_path, cluster_edges, cluster_endpoints = run_course_opt(cities, centers, niter=500)
    
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
