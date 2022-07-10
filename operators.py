# -*- coding: utf-8 -*-
import numpy as np

from data import calc_euclidean_dist


def shit_rng2(low, high, excludes):
    
    r2 = -1

    while r2 == -1:
        rt = np.random.randint(low, high)
            
        if rt not in excludes:
            r2 = rt
        
    return r2


def shit_rng(low, high, r1, diff, n):
        
    r2a = np.maximum(low, np.random.randint(low - diff - 1, r1 - diff, n))

    r2b = np.minimum(high - 1, np.random.randint(r1 + diff, high + diff + 1, n))

    ri = np.random.randint(0, 1, n)

    r = r2a * (ri) - r2b * (ri-1)
    
    return r


def calc_dist(tour, d):
    return sum(d[tour[i], tour[i+1]] for i in range(0, len(tour) - 1))


def calc_relocate_change(tour, d, t_id, r_id):
    # calculate the change in distance from a replacement move

    # previous edge lengths:

    # d(n[t-1], n[t]) + d(n[t], n[t+1])

    # d(n[r], n[r+1])

    # new edges:

    # d(n[t-1], d[t+1])

    # d(n[r-1], n[t]) + d(n[t], n[r+1])
    
    n_tid = tour[t_id]
    
    n_tid_m = tour[t_id - 1]
    
    n_tid_p = tour[t_id + 1]
    
    n_rid = tour[r_id]
    
    n_rid_p = tour[r_id + 1]

    before = (d[n_tid_m, n_tid]
              + d[n_tid, n_tid_p]
              + d[n_rid, n_rid_p])

    after = (d[n_tid_m, n_tid_p]
             + d[n_rid, n_tid]
             + d[n_tid, n_rid_p])

    return after - before


def relocate_node(tour, t_id, r_id):

    if t_id > r_id:
        r_id += 1
        
    removed = tour.pop(t_id)

    tour.insert(r_id, removed)

    return tour


def relocate(tour, t_id, r_id, d):
    if calc_relocate_change(tour, d, t_id, r_id) < 0:
        tour = relocate_node(tour, t_id, r_id)
    return tour


def gen_edges_2opt(tour, n_id1, n_id2):

    n1, n2 = tour[n_id1], tour[n_id2]
    
    e1 = (n1, tour[n_id1 + 1])
    
    e2 = (n2, tour[n_id2 + 1])

    return e1, e2


def make_2opt(tour, e1, e2):
    
    idx_start = tour.index(e1[1])
    
    idx_end = tour.index(e2[1])
        
    tour[idx_start:idx_end] = reversed(tour[idx_start:idx_end])
    
    return tour


def two_opt(tour, d, n_id1, n_id2):
    
    # 2-opt condisers edges e1 = (a, b) & e2 = (x, y)
    
    # valid alternative (to maintain a cycle) is c1 = (a, x) and c2 = (b, y)
    
    # swap edges iff d[e1] + d[e2] > d[c1] + d[c2]
    
    e1, e2 = gen_edges_2opt(tour, n_id1, n_id2)
        
    c1 = (e1[0], e2[0])

    c2 = (e1[1], e2[1])

    old = d[e1] + d[e2]

    diff = (d[c1] + d[c2]) - old
    
    if diff < 0:
        tour = make_2opt(tour, e1, e2)

    return tour


def local_search(tour, X, Y, niter: int=5):

    tour = list(tour)
    
    n = len(tour)

    d = calc_euclidean_dist(X, Y)

    # start and end nodes cannot be moved
    #candidates = list(range(1, n - 1))
    
    # random numbers for relocate
    t_id_arr = np.random.randint(1, n - 1, niter)
    
    r_id_arr = shit_rng(1, n - 1, t_id_arr, 1, niter)
    
    # random numbers for two_opt
    n_id1_arr = np.random.randint(1, n - 2, niter)
    
    n_id2_arr = shit_rng(1, n - 2, n_id1_arr, 2, niter)

    for i in range(0, niter):
        t_id = t_id_arr[i]
        
        r_id = r_id_arr[i]
                
        n_id1 = n_id1_arr[i]
        
        n_id2 = n_id2_arr[i]
        
        # relocate can generate unique solutions other ops cannot.
        tour = relocate(tour, t_id, r_id, d)
        
        tour = two_opt(tour, d, n_id1, n_id2)
        
        # if there is time, add 3 opt.

    tour = np.array(tour)
    return tour

if __name__ == '__main__':
    nodes = [0, 1, 2, 3, 4, 5, 6]

    X = [1, 4, 2, 6, 8, 22, 0]

    Y = [100, 50, 20, 200, 300, 30, 500]

    n = len(nodes)

    d = calc_euclidean_dist(X, Y)

    tour = nodes.copy()

    sedges = [(nodes[i], nodes[i+1]) for i in range(0, n - 1)]

    ncheck = 5
    
    tour = local_search(tour, X, Y, 50000)
