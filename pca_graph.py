#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def get_pca_graph(samples, labels, out_file, plot_negative=True, 
        title=None, caption=None):
    pca = PCA(n_components=2)
    points = pca.fit_transform(samples)

    positive_x = [p for p, l in zip(points[:,0], labels) if l == 1]
    positive_y = [p for p, l in zip(points[:,1], labels) if l == 1]
    negative_x = [p for p, l in zip(points[:,0], labels) if l == 0]
    negative_y = [p for p, l in zip(points[:,1], labels) if l == 0]

    
    legend_points = ()
    legend_labels = ()

    if plot_negative:
        negative = plt.scatter(negative_x, negative_y, color='blue', alpha=.5)
        legend_points += (negative,)
        legend_labels += ('Clean',)
    
    positive = plt.scatter(positive_x, positive_y, color='red', alpha=.5)
    legend_points += (positive,)
    legend_labels += ('Insult',)

    if title:
        plt.title(title)
    plt.legend(legend_points, legend_labels)
    plt.savefig(out_file)
    plt.clf()
