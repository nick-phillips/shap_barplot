# Create a barplot of most important features from SHAP
# Features are ordered from top to bottom by decreasing 
# importance. Color indicates the assocation with positive (1)
# classification. Only used in binary classification.
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import shap

def _get_feature_importance(X, shap_values, col_names, num_features=20):
    """ Compute feature importance and effect with respect to feature
    mean from shap_values and X

    Args:
        shap_values {array} -- shap values of test set, shape (num_examples, num_features)
        X {array} -- data, shape (num_examples, num_features)
        col_names {list} -- Column names of X
        num_features {int} -- number of features to plot
    """
    # Determine median value for each feature
    median_value = np.median(X, axis=0)
    # Keep where feature contributes positives to classification (shap_values > 0)
    # and the value is greater than the median value
    pos_effect = (shap_values >= 0) * (X >= median_value) * (X)
    # Percentage high value features contrinute to positive or negative effects
    percent_pos_effect = (pos_effect > 0).sum(axis=0) / pos_effect.shape[0]
    color = percent_pos_effect
    # compute feature importance
    fi_shap = abs(shap_values).sum(0)
    # Normalize
    fi_shap = fi_shap / fi_shap.sum() * 100
    ind = (-fi_shap).argsort()[:num_features]
    return fi_shap[ind], color[ind], col_names[ind]

def shap_barplot(X, shap_values, column_names, num_features=20, fig=None, figsize=(10,8)):
    """ Shap bar plot.

    Args:
        shap_values (np.array): SHAP values from package
        X (np.array): data, shape (num_examples, num_features)
        column_names (list): Column names of X
        ax (pyplot axes, optional): Provided axes to plot. Defaults to None.
    """
    # set plot formatting
    sns.set_style('dark')
    sns.set_context('paper')
    rcParams.update({'figure.autolayout': True})

    # get feature importance, colors, colnames of top features
    feature_importance, color, colnames = _get_feature_importance(X, shap_values, column_names, num_features=num_features)

    # format colormap
    scaled_colors = (color - min(color)) / (max(color) - min(color))
    color_list = [(.263, .557, .749), (1, 1, 1), (.722, .059, .047)]  # Blue -> Red
    cmap = LinearSegmentedColormap.from_list('plot_map', color_list, N=100)

    if not fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = fig.gca()

    # plot barplot
    sns.barplot(x=feature_importance, y=colnames, ax=ax, \
        palette=cmap(scaled_colors), orient='h')
    ax.set_xlabel("Feature Importance (%)")
    ax.set_ylabel('Feature Names')

    # plot colorbar
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ticks=[0, 1], aspect=1000)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature Value", size=12, labelpad=0)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - .9) * 20)
    
    # get results dataframe
    results_df = pd.DataFrame({"Feature Names": colnames, "Feature Importance": feature_importance, "Relative Values": scaled_colors})

    return fig, results_df
