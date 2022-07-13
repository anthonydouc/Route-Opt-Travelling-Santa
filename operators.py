# -*- coding: utf-8 -*-
import numpy as np

from data import calc_euclidean_dist


def calc_dist(tour, d):
    return sum(d[tour[i], tour[i+1]] for i in range(0, len(tour) - 1))


def get_three_segs_rand(tour: list, niter: int, rng):
    
    n = len(tour)

    ri = rng.integers(1, n - 1 - 2 - 2, niter)

    rj = rng.integers(ri + 2, n - 1 - 2, niter)
        
    rk = rng.integers(rj + 2, n - 1, niter)

    return list(zip(ri, rj, rk))


def three_opt(tour, dist, seg):
    
    for i,j,k in seg:

        move = check_distances(tour, dist, i, j, k)

        if move is not None:
            tour = make_3opt(tour, move, i, j, k)
            
    return tour


def make_3opt(tour, move, i, j, k):
    # https://isd.ktu.lt/it2011/material/Proceedings/1_AI_5.pdf
    # unique 3- opt alternatives:
    # 1. AB -> AD, CD -> CF, EF -> EB: (a, d), (c, f), (e, b) [3-opt]
    # 2. AB -> AC, CD -> DF, EF -> BE: (a, c), (b, e), (d, f) [3-opt]
    # 3. AB -> AD, CD -> CE, EF -> BF: (a, d), (c, e), (b,f) [3-opt]
    # 4. AB -> AE, CD -> DB, EF -> CF: (a, e), (c, f), (d, b) [3-opt]

    # 5. AB -> AC, CD -> CB (reverse BC) : (a, c), (b, d), (e, f) [2-opt]
    # 6. CD -> CE, DE -> DF (reverse DE): (a, b), (c, e), (d, f) [2-opt]
    # 7. AB -> AE, EF -> FB (reverse BCDE): (a, e), (c, d), (b, f) [2-opt]

    if move == 1:
        tour[i+1:k+1] = tour[j+1:k+1] + tour[i+1:j+1] # swap BC with DE
    elif move == 2:
        tour[i+1:k+1] = tour[j:i:-1] + tour[k:j:-1] # reverse BC & DE
    elif move == 3:
        tour[i+1:k+1] = tour[j+1:k+1] + tour[j:i:-1] 
    elif move == 4:
        tour[i+1:k+1] = tour[k:j:-1] + tour[i+1:j+1] # swap BC with ED

    return tour


def check_distances(tour, dist, i, j, k):

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
        
    ne1 = (e1[0], e2[0])

    ne2 = (e1[1], e2[1])

    diff = (d[ne1] + d[ne2]) - (d[e1] + d[e2])
    
    if diff < 0:
        tour = make_2opt(tour, e1, e2)
    return tour

 
def local_search(tour, X, Y, niter: int=5):
    
    rng = np.random.default_rng(0)

    tour = list(tour)
    
    n = len(tour)

    d = calc_euclidean_dist(X, Y)

    # random numbers for relocate.
    t_id_arr = rng.integers(1, n - 2, niter)
        
    r_id_arr = rng.integers(t_id_arr + 1, n - 1, niter)
            
    # random numbers for two_opt
    n_id1_arr = rng.integers(1, n - 2 - 2, niter)
    
    n_id2_arr = rng.integers(n_id1_arr + 2, n - 2, niter)
    
    segs = get_three_segs_rand(tour, niter, rng)
       
    for i in range(0, niter):
        t_id, r_id = t_id_arr[i], r_id_arr[i]
                        
        n_id1, n_id2 = n_id1_arr[i], n_id2_arr[i]
                
        tour = relocate(tour, t_id, r_id, d)
        
        tour = two_opt(tour, d, n_id1, n_id2)
        
        tour = three_opt(tour, d, [segs[i]])

    tour = np.array(tour)
    return tour

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
