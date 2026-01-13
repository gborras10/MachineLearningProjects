"""
    This module includes plotting tools to visualize machine learning problems

    Author: <alberto.suarez@uam.es>
"""

import numpy as np
import matplotlib.pyplot as plt

def label_figure(ax, X, y, indices_features=(0, 1), fontsize=14):
    
    class_labels = np.unique(y)
    n_instances, n_features = np.shape(X)
    
    ax.set_xlabel('$X_{}$'.format(indices_features[0]), fontsize=fontsize)
    ax.set_ylabel('$X_{}$'.format(indices_features[1]), fontsize=fontsize)

    ax.set_title('# instances = {},  # features  = {} \n Class labels = {}'.format(
        n_instances, n_features, class_labels), fontsize=int(fontsize*1.2)) 


def plot_dataset_2D(X, y, indices_features=(0, 1), ax=None, fontsize=14):
    """ Plot in 2 D the attribute space and the class labels.
        The class labels are indicated by color of the plotted points.
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    ax.scatter(X[:, indices_features[0]],
               X[:, indices_features[1]],
               marker='o', c=y, s=100, edgecolor='k')
    
    label_figure(ax, X, y, indices_features, fontsize)

    return ax


def plot_2D_decision_regions(X, y, 
                             decision_function, 
                             decision_levels=None,
                             cmap=plt.cm.bwr, 
                             alpha_light=0.1, alpha_dark=1.0,
                             ax=None, fontsize=14):    
    """ Plot the decision boundary for a classification problem with two features """
    
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max] x [y_min, y_max].
    
    x_plot = X[:, 0]
    y_plot = X[:, 1]


    
    # Plot the class label predictions with a color encoding
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    
    # Plot the training points
   
    _ = ax.scatter(X[:, 0], X[:, 1], s=30, c=y, 
                    alpha=alpha_dark, cmap=cmap, edgecolors='k')

   # Decision regions plot
   
    x_plot_min, x_plot_max  = ax.get_xlim() 
    y_plot_min, y_plot_max  = ax.get_ylim()
    
    n_plot = 200
    xx_plot = np.linspace(x_plot_min, x_plot_max, n_plot)
    yy_plot = np.linspace(y_plot_min, y_plot_max, n_plot)

    XX_plot, YY_plot = np.meshgrid(xx_plot, yy_plot)
    
    ZZ_plot = decision_function(np.c_[XX_plot.ravel(), YY_plot.ravel()])
    ZZ_plot = ZZ_plot.reshape(XX_plot.shape)

    _ = ax.imshow(ZZ_plot, interpolation='nearest',
                   extent=(x_plot_min, x_plot_max, y_plot_min, y_plot_max), 
                   aspect='auto', origin='lower', 
                   alpha=alpha_light,  cmap=cmap)

   # Contour plot


    if decision_levels == 'auto':
        min_Z = np.min(ZZ_plot)
        max_Z = np.max(ZZ_plot)
        mid_Z = 0.5*(min_Z + max_Z)

        decision_levels = [min_Z, mid_Z, max_Z]

    if decision_levels is not None:
        contours = ax.contour(XX_plot, YY_plot, ZZ_plot, 
                              levels=decision_levels, colors='k',
                              linewidths=1, linestyles='dashed')

    label_figure(ax, X, y, fontsize=fontsize)
   
    return ax

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


