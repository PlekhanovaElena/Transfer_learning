from pandas import Series, DataFrame
import multiprocessing as mp
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def get_grid_searcher(estimator, params, X_train, y_train, scoring='f1', cv=5, n_jobs=2, outprint = True):
    grid_searcher = GridSearchCV(estimator, params, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid_searcher.fit(X_train, y_train)
    if outprint:
        print(grid_searcher.best_score_)
        print(grid_searcher.best_estimator_)

    return grid_searcher


def plot_quality(grid_searcher, log_scale=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    means = []
    stds = []
    param_name = list(grid_searcher.grid_scores_[0].parameters)[0]
    for score in grid_searcher.grid_scores_:
        means.append(np.mean(score.cv_validation_scores))
        stds.append(np.sqrt(np.var(score.cv_validation_scores)))

    means = np.array(means)
    stds = np.array(stds)
    params = grid_searcher.param_grid[param_name]

    if isinstance(params[0], str):
        params = range(len(params))

    ax.grid(True)
    ax.plot(params, means)

    lower = means - stds
    upper = means + stds

    ax.plot(params, lower, 'b--')
    ax.plot(params, upper, 'b--')
    ax.fill_between(params, lower, upper, alpha=0.1, color="b")
    ax.set_ylabel('CV score')
    ax.set_xlabel(param_name)
    ax.axhline(np.max(means), linestyle='--', color='r')
    if log_scale:
        ax.set_xscale('log')

def map_parallel(function, arr, threads_num=50):
    with mp.Pool(threads_num) as p: res = p.map(function, arr)
    if isinstance(arr, Series):
        return Series(res, arr.index)
    
    return res
