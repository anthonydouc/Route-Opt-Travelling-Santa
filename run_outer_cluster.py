# -*- coding: utf-8 -*-

from data import (get_data, calc_euclidean_dist, create_submission,
                  get_node_positions, save_tour, load_tour)

from lpalgorithms import get_path_mtz

from algorithms import nearest_neighbour, get_endpoints, opt_cluster

from operators import local_search 

n = 197769

cities, centers = get_data(used_saved=True, ncluster=500)

pX, pY = get_node_positions(cities)

nodes = centers.index.values

X, Y = centers['X'], centers['Y']

cluster_path, cluster_edges = nearest_neighbour(nodes, X, Y, 0, 0)

cluster_path = local_search(cluster_path, X, Y, 10000000)

cluster_endpoints = get_endpoints(cities, cluster_edges)

path = []

# can we parallelise this
for cluster in cluster_path[:-1]:
    print(cluster)
    c_path = opt_cluster(cities, cluster_endpoints, cluster_edges, cluster)

    path += list(c_path)

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


