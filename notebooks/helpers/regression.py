from typing import (
    List,
    Tuple,
)
from itertools import (
    product,
    cycle,
)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.formula.api import ols

from .load_data import (
    get_data_with_cpu,
    trim_experiment,
)


COLORS = ['darkturquoise', 'yellowgreen', 'tomato', 'gold']
sns.set_theme(style="whitegrid")


def fit_regression(data, formula):
    data = sm.add_constant(data)
    model = ols(data=data, formula=formula)
    results = model.fit()
    return results


def draw_regression_graph(results, df, metric_name='app_latency', variable='instances_n', alpha=0.05):
    x = df[variable]
    y = df[metric_name]

    prstd, iv_l, iv_u = wls_prediction_std(results, alpha=alpha)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'o', label="data")
    ax.plot(x, results.fittedvalues, 'r--.', label="OLS")
    ax.plot(x, iv_u, 'r--')
    ax.plot(x, iv_l, 'r--')

    ax.set_xlabel(variable)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} regression')
    ax.legend(loc='best')


def boxplot_two(var_1: str, var_2: str, df: pd.DataFrame, figsize: Tuple[int, int], title: str, ylab_1='', ylab_2=''):
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False, sharey=False, figsize=figsize)
    ax1 = sns.boxplot(x='instances_n', y=var_1, data=df, ax=ax1, color='yellowgreen')
    if title:
        ax1.set_title(title)
    if ylab_1:
        ax1.set_ylabel(ylab_1)
    
    sns.boxplot(x='instances_n', y=var_2, data=df, ax=ax2, color='tomato')
    if ylab_2:
        ax2.set_ylabel(ylab_2)    


def boxplot_grid(var_names: List[List[str]], dfs: List[List[pd.DataFrame]], figsize: Tuple[int, int], titles: List[List[str]], ylabels: List[List[str]], suptitle: str=None):
    n = len(var_names)
    m = len(var_names[0])
    fig, axes = plt.subplots(nrows=n, ncols=m, sharex=False, sharey=False, figsize=figsize)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, y=0.91)
    
    for (i, j), color in zip(product(range(n), range(m)), cycle(COLORS)):
        ax = sns.boxplot(x='instances_n', y=var_names[i][j], data=dfs[i][j], ax=axes[i, j], color=color)
        if titles:
            try:
                ax.set_title(titles[i][j])
            except IndexError:
                ax.set_title('')
        if ylabels:
            try:
                ax.set_ylabel(ylabels[i][j])
            except IndexError:
                ax.set_ylabel('')


def get_coeff(experiment_path: str, instances_n: int, trim_len: int, target: str):
    df = get_data_with_cpu(experiment_path, instances_n=instances_n, cpu_window=30)
    df = trim_experiment(df, trim_len)
    results = fit_regression(data=df, formula=f'{target} ~ instances_n')
    return results.params[1]
