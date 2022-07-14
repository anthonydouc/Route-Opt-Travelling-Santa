# -*- coding: utf-8 -*-
import numpy as np
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.layouts import row

from routeopt import (get_data, get_node_positions, load_route,
                      get_node_primality, calc_path_distance_prime)


def create_bk_plot(**kwargs):
    fig = figure(**kwargs)
    fig = plot_format(fig)
    return fig


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

path = load_route('route')

f1 = create_bk_plot(output_backend='webgl',
                    height=700, width=1000,
                    x_range=(min(pX), max(pX)),
                    y_range=(min(pY), max(pY)))


f1.line(x=pX[path], y=pY[path])

f1.scatter(x=[pX[0]], y=[pY[0]], size=40 , color='#fb8500', marker='star')

show(f1)
