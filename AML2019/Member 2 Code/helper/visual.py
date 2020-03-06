import math
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from matplotlib import pyplot as plt


def grid_search_plot(grid_search, param_name, ax=None, fit_time=False):
    assert (isinstance(grid_search, pd.DataFrame)
            or isinstance(grid_search, GridSearchCV)
            ), 'grid_search must be of GridSearchCV or pandas.DataFrame'
    if ax is None:
        ax = plt.gca()
    if not isinstance(grid_search, pd.DataFrame):
        df = pd.DataFrame(grid_search.cv_results_)

    df.sort_values(by='rank_test_score')
    to_plot = 'fit_time' if fit_time else 'test_score'

    means = df['mean_{}'.format(to_plot)]
    stds = df['std_{}'.format(to_plot)]
    params = df['param_' + param_name]

    ax.errorbar(params, means, yerr=stds)
    ax.set_xlabel(param_name)

    return ax


def plot_grid_search_all(grid_search, fname=None, fit_time=False, title=None):

    param_all = list(grid_search.param_grid.keys())
    nrows = math.ceil(math.sqrt(len(param_all)))
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows,
                             figsize=(3*len(param_all), 3*len(param_all)))
    if nrows == 1:
        grid_search_plot(grid_search, param_all[0], axes, fit_time=fit_time)
    else:
        for ax, p in zip(axes.flat, param_all):
            grid_search_plot(grid_search, p, ax, fit_time=fit_time)

    plt.tight_layout()

    if not title:
        title = '{} vs param'.format(
            'Score') if not fit_time else '{} vs param'.format('fit_time')
    st = fig.suptitle(title, fontsize="x-large")
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)

    if fname:
        plt.savefig(fname, dpi=200)

    plt.close()
    return fig
