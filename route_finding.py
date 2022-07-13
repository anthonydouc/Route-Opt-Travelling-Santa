# -*- coding: utf-8 -*-
import numpy as np

from data import calc_euclidean_dist


def calc_dist(tour, d):
    return sum(d[tour[i], tour[i+1]] for i in range(0, len(tour) - 1))


def get_edges_from_path(path):
    return [(path[i], path[i+1]) for i in range(0, len(path) - 1)]


def relocate(tour: list, t_id: int, r_id: int, dist: np.array): 
    '''
    Swap the position of the node tour[t_id] from t_id to r_id.
    
    iff delta_distance < 0
    
    where delta_distance = [ d(n[t-1], n[t]) + d(n[t], n[t+1]) + d(n[r], n[r+1]) ]
                          - [d(n[t-1], d[t+1]) + d(n[r-1], n[t]) + d(n[t], n[r+1])]

    See https://www.mdpi.com/2076-3417/9/19/3985 for examples.
    '''

    n_tid = tour[t_id]
    
    n_tid_m = tour[t_id - 1]
    
    n_tid_p = tour[t_id + 1]
    
    n_rid = tour[r_id]
    
    n_rid_p = tour[r_id + 1]

  
    dist_before = (dist[n_tid_m, n_tid]
                   + dist[n_tid, n_tid_p]
                   + dist[n_rid, n_rid_p])
    
    dist_after = (dist[n_tid_m, n_tid_p]
                  + dist[n_rid, n_tid]
                  + dist[n_tid, n_rid_p])
    
    if dist_after - dist_before < 0:

        if t_id > r_id:
          r_id += 1
         
        removed = tour.pop(t_id)
        
        tour.insert(r_id, removed)

    return tour


def get_relocate_indices(rng, n_nodes, n_samples):
    '''
    Randomly generates two arrays of valid tour edges for node relocation.
    '''
    
    t_id_arr = rng.integers(1, n_nodes - 2, n_samples)
        
    r_id_arr = rng.integers(t_id_arr + 1, n_nodes - 1, n_samples)
    
    return t_id_arr, r_id_arr


def two_permute(tour: list, dist: np.array, s1: int, s2: int) -> list:
    '''
    Swaps the connection between two edges:
        e1 = (tour[s1], tour[s1 + 1]) &
        e2 = (tour[s2], tour[s2+1])
        
    and their valid alternative (that maintains a hamiltonian cycle):
        c1 = (tour[s1], tour[s2])
        c2 = (tour[s1 + 1], tour[s2 + 1])
    
    iff dist[e1] + dist[e2] - dist[c1] + dist[c2] < 0
    
    '''
    
    e1, e2 = (tour[s1], tour[s1 + 1]), (tour[s2], tour[s2 + 1])
            
    c1, c2 = (tour[s1], tour[s2]), (tour[s1 + 1], tour[s2 + 1])
    
    if (dist[c1] + dist[c2]) - (dist[e1] + dist[e2]) < 0:
        tour[s1+1:s2+1] = reversed(tour[s1+1:s2+1])
        
    return tour


def get_two_permute_indices(rng, n_nodes: int, n_samples: int) -> (np.array, np.array):
    '''
    Randomly generates two arrays of valid tour edges for two opt style 
    permutation. 
    '''
    
    s1_arr = rng.integers(1, n_nodes - 2 - 2, n_samples)
    
    s2_arr = rng.integers(s1_arr + 2, n_nodes - 2, n_samples)
    
    return s1_arr, s2_arr


def three_permute(tour: list, dist: np.array, i: int, j: int, k: int):
    '''
    Swaps the linkages between the nodes in the three edges:
        (tour[i], tour[i+1]), (tour[j], tour[j+1]), (tour[k], tour[k+1])
    
    with a valid alternative iff. it leads to a decrease in tour length.
    
    There are four unique 3-opt alternatives:
    1. AB -> AD, CD -> CF, EF -> EB: (a, d), (c, f), (e, b)
    2. AB -> AC, CD -> DF, EF -> BE: (a, c), (b, e), (d, f)
    3. AB -> AD, CD -> CE, EF -> BF: (a, d), (c, e), (b,f) 
    4. AB -> AE, CD -> DB, EF -> CF: (a, e), (c, f), (d, b) 
    
    The remainder can be performed by 2-opt permutations.
    
    See https://isd.ktu.lt/it2011/material/Proceedings/1_AI_5.pdf.
    '''
    
    move = check_three_permute_distances(tour, dist, i, j, k)

    if move is None:
        pass
    elif move == 1:
        tour[i+1:k+1] = tour[j+1:k+1] + tour[i+1:j+1]
    elif move == 2:
        tour[i+1:k+1] = tour[j:i:-1] + tour[k:j:-1]
    elif move == 3:
        tour[i+1:k+1] = tour[j+1:k+1] + tour[j:i:-1] 
    elif move == 4:
        tour[i+1:k+1] = tour[k:j:-1] + tour[i+1:j+1]
            
    return tour


def check_three_permute_distances(tour: list, dist: np.array, i: int, j: int, k: int) -> int:
    '''
    Calculates the change in distance from all valid unique three edge
    permutations for a hamiltonian cycle.
    '''

    a, b, c, d, e, f = tour[i], tour[i+1], tour[j], tour[j+1], tour[k], tour[k+1]

    d0 = dist[a, b] + dist[c, d] + dist[e, f]

    d1 = dist[a, d] + dist[c, f] + dist[e, b]

    d2 = dist[a, c] + dist[b, e] + dist[d, f]

    d3 = dist[a, d] + dist[c, e] + dist[b, f]

    d4 = dist[a, e] + dist[c, f] + dist[d, b]

    diffs = [d1 - d0, d2 - d0, d3 - d0, d4 - d0]

    mindiff = min(diffs)
    
    if mindiff < 0:
        return diffs.index(mindiff) + 1
    else:
        return None


def get_three_permute_indices(rng, n_nodes: int, n_samples: int):
    '''
    Randomly generates arrays of valid tour edges for three opt style 
    permutation. 
    '''
    
    ri = rng.integers(1, n - 1 - 2 - 2, n_samples)

    rj = rng.integers(ri + 2, n - 1 - 2, n_samples)
        
    rk = rng.integers(rj + 2, n - 1, n_samples)

    return ri, rj, rk


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


def local_search(tour, X, Y, niter: int=5):
    
    rng = np.random.default_rng(0)

    tour = list(tour)
    
    n = len(tour)
    
    # distance matrix between all nodes in the tour
    distances = calc_euclidean_dist(X, Y)

    # random numbers for relocate.
    t_id_arr, r_id_arr = get_relocate_indices(rng, n, niter)
    
    # random numbers for two permute
    s1_arr, s2_arr = get_two_permute_indices(rng, n, niter)
    
    # random numbers for three permute
    ri_arr, rj_arr, rk_arr = get_three_permute_indices(rng, n, niter)
       
    for i in range(0, niter):
        t_id, r_id = t_id_arr[i], r_id_arr[i]
                                        
        tour = relocate(tour, t_id, r_id, distances)
        
        tour = two_permute(tour, distances, s1_arr[i], s2_arr[i])
        
        tour = three_permute(tour, distances, ri_arr[i], rj_arr[i], rk_arr[i])

    tour = np.array(tour)
    return tour


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

if __name__ == '__main__':
    n = 100
    
    nodes = list(range(0, n))

    X = np.random.randint(1, 50, n)

    Y = np.random.randint(20, 500, n)

    d = calc_euclidean_dist(X, Y)

    tour = nodes.copy()

    sedges = [(nodes[i], nodes[i+1]) for i in range(0, n - 1)]

    ncheck = 5
    
    tour = local_search(tour, X, Y, 100000)
