from datetime import datetime
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.formula.api import ols


def get_experiments_data(experiment_name: str, experiments_path: str):
    for exp_dir in sorted(glob.glob(f'{experiments_path}/{experiment_name}_n*')):
        exp_name = exp_dir.rpartition("/")[-1]
        csv_path = f'{exp_dir}/VM_runtime_app_{exp_name}.csv'
        df = pd.read_csv(csv_path, sep=',', skiprows=28)
        yield exp_name, df.get(['time', 'time_h', 'ai_name', 'app_throughput', 'app_latency'])

        
def get_attach_indexes(df: pd.DataFrame, instances_n=6):
    res = []
    for i in range(1, instances_n+1):
        ai_name = f'ai_{i}' 
        res.append((df['ai_name'] == ai_name).argmax())
    res.append(len(df))
    return res


def calc_means(df: pd.DataFrame, instances_n=6):
    attach_indexes = get_attach_indexes(df, instances_n=instances_n)
    
    means = []
    for i, (a, b) in enumerate(zip(attach_indexes, attach_indexes[1:])):
        df_t = df.iloc[a:b]
        df_t = df_t[df_t['ai_name'] == 'ai_1'].mean()
        means.append(df_t)
    
    res = pd.DataFrame(means)
    res = res.drop('time', axis=1)
    res.insert(loc=0, column='instances_n', value=list(range(1, instances_n + 1)))
    return res

def add_instances_n(df, instances_n=6):
    df_r = df.copy()
    attach_indexes = get_attach_indexes(df, instances_n=instances_n)
    
    for i, (a, b) in enumerate(zip(attach_indexes, attach_indexes[1:])):
        df_r.at[a:b, 'instances_n'] = i + 1

    return df_r


def get_means(experiment_name: str, experiments_path: str, instances_n=6):
    for exp_name, df in get_experiments_data(experiment_name, experiments_path):
        df = df.get(['time', 'ai_name', 'app_throughput', 'app_latency'])
        df_means = calc_means(df, instances_n=instances_n)
        yield exp_name, df_means

        
def get_means_merged(experiment_name: str, experiments_path: str, instances_n=6):
    res = pd.DataFrame(columns=['instances_n', 'app_throughput', 'app_latency'])
    
    for exp_name, df in get_experiments_data(experiment_name, experiments_path):
        df = df.get(['time', 'ai_name', 'app_throughput', 'app_latency'])
        df_means = calc_means(df)
        res = res.append(df_means)
    
    res.index = range(len(res))
    return res


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


def get_attach_times(df: pd.DataFrame, ai_n=6):
    res = []
    for i in range(1, ai_n+1):
        ai_name = f'ai_{i}' 
        t = df['time'][(df['ai_name'] == ai_name).argmax()]
        res.append(t)
    res.append(df['time'][len(df) - 1] + 1)
    return res
        
        
def get_means_from_interval(df, a, b):
    df_t = df[(a <= df['time']) & (df['time'] < b)]
    return df_t.mean()


def get_means_with_cpu(experiment_name: str, experiments_path: str, instances_n=6):
    for exp_dir in sorted(glob.glob(f'{experiments_path}/{experiment_name}_n*')):
        # cbtool data
        exp_name = exp_dir.rpartition("/")[-1]
        csv_path = f'{exp_dir}/VM_runtime_app_{exp_name}.csv'
        df = pd.read_csv(csv_path, sep=',', skiprows=28)
        df = df.get(['time', 'ai_name', 'app_throughput', 'app_latency'])

        # cpu data
        cpu_csv_path = f'{exp_dir}/resources/metric_node_baati_cpu.csv'
        df_cpu = pd.read_csv(cpu_csv_path)
        df_cpu = df_cpu.get(['time', 'value'])
        df_cpu['time'] = df_cpu['time'].apply(lambda x: int(str(x)[:10]))
        time_diff = df_cpu['time'][0] - df['time'][0]
        df_cpu['time'] = df_cpu['time'].apply(lambda x: x - time_diff)

        attach_times = get_attach_times(df, ai_n=instances_n)
        df = df[df['ai_name'] == 'ai_1']

        means = []
        for i, (a, b) in enumerate(zip(attach_times, attach_times[1:])):
            df_t = get_means_from_interval(df, a, b)
            df_cpu_t = get_means_from_interval(df_cpu, a, b)
            df_t['cpu'] = df_cpu_t['value']
            means.append(df_t)

        res = pd.DataFrame(means)
        res.insert(loc=0, column='instances_n', value=list(range(1, instances_n + 1)))
        yield exp_name, res


def convert_from_time_str(time_str: str):
    parsed_time = datetime.strptime(time_str + ' +0000', '%m/%d/%Y %I:%M:%S %p %Z %z')
    return int(datetime.timestamp(parsed_time))


def convert_from_padded(padded_str: str):
    return int(str(padded_str)[:10])


def get_data_with_metrics(experiment_name: str, experiments_path: str, instances_n=6, max_time_diff=5):
    for exp_dir in sorted(glob.glob(f'{experiments_path}/{experiment_name}_n*')):
        # cbtool data
        exp_name = exp_dir.rpartition("/")[-1]
        csv_path = f'{exp_dir}/VM_runtime_app_{exp_name}.csv'
        df = pd.read_csv(csv_path, sep=',', skiprows=28)
        df = df.get(['time_h', 'ai_name', 'app_throughput', 'app_latency'])
        df['time_h'] = df['time_h'].apply(convert_from_time_str)
        df = df.rename(columns={'time_h': 'time'})
        df = add_instances_n(df, instances_n=instances_n)
        df = df[df['ai_name'] == 'ai_1']

        # cpu data
        cpu_csv_path = f'{exp_dir}/resources/metric_node_baati_cpu.csv'
        df_cpu = pd.read_csv(cpu_csv_path)
        df_cpu = df_cpu.get(['time', 'value'])
        df_cpu['time'] = df_cpu['time'].apply(convert_from_padded)
        df_cpu = df_cpu.rename(columns={'value': 'cpu'})
        df_cpu['cpu'] = df_cpu['cpu'].astype(float)

        # memory data
        mem_csv_path = f'{exp_dir}/resources/metric_node_baati_memory.csv'
        df_mem = pd.read_csv(mem_csv_path)
        df_mem = df_mem.get(['time', 'value'])
        df_cpu['memory'] = df_mem['value']

        df_res = pd.DataFrame(columns=['cbtool_time', 'cpu_time', 'app_latency', 'app_throughput', 'cpu', 'memory', 'instances_n'])

        # merge dataframes
        for i, row in df_cpu.iterrows():
            time = row['time']
            closest_row = df.iloc[(df['time'] - time).abs().argsort()[0]]

            if abs(closest_row['time'] - time) > max_time_diff:
                continue

            df_res = df_res.append({
                'cpu_time': str(int(time)),
                'cbtool_time': str(int(closest_row['time'])),
                'app_throughput': closest_row['app_throughput'],
                'app_latency': closest_row['app_latency'],
                'cpu': row['cpu'],
                'memory': row['memory'],
                'instances_n': closest_row['instances_n'],
            }, ignore_index=True)

        yield exp_name, df_res
