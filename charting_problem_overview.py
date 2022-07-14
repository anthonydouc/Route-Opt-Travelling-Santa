# -*- coding: utf-8 -*-
from bokeh.plotting import figure
from bokeh.io import show

from routeopt import get_data, get_node_positions

def plot_format(p):
        
    p.xaxis.visible = False
    p.yaxis.visible = False 
    p.xgrid.visible = False
    p.ygrid.visible = False

    p.outline_line_color = None
    p.border_fill_color = None
    p.border_fill_alpha = 0
    
    return p

cities, centers = get_data(used_saved=True, ncluster=500)

pX, pY = get_node_positions(cities)

fig = figure(output_backend='webgl',
             height=800, width=1200, 
             x_range=(min(pX), max(pX)),
             y_range=(min(pY), max(pY)))

fig = plot_format(fig)

fig.scatter(x=pX, y=pY, size=0.0001, color='#0077b6')

fig.scatter(x=[pX[0]], y=[pY[0]], size=40 , color='#fb8500', marker='star')

show(fig)
