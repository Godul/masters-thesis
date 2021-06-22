from bisect import (
    bisect_left,
    bisect_right,
)
from datetime import (
    datetime,
    timedelta,
)
import glob
from itertools import chain
import re

import numpy as np
import pandas as pd


def get_experiments_paths(experiment_name: str, experiments_path: str):
    return list(sorted(glob.glob(f'{experiments_path}/{experiment_name}_m*')))


def convert_from_time_str(time_str: str):
    parsed_time = datetime.strptime(time_str + ' +0000', '%m/%d/%Y %I:%M:%S %p %Z %z')
    return int(datetime.timestamp(parsed_time))


def convert_from_padded(padded_str: str):
    "Convert timestamp from weird padded format in our .csv"
    return int(str(padded_str)[:10])


def get_attach_indexes(df: pd.DataFrame, instances_n: int):
    res = []
    for i in range(1, instances_n + 1):
        ai_name = f'ai_{i}'
        res.append((df['ai_name'] == ai_name).argmax())
    res.append(len(df))
    return res


def add_instances_n_old(df: pd.DataFrame, instances_n: int):
    df_r = df.copy()
    attach_indexes = get_attach_indexes(df, instances_n=instances_n)
    
    for i, (a, b) in enumerate(zip(attach_indexes, attach_indexes[1:])):
        df_r.at[a:b, 'instances_n'] = i + 1

    return df_r


def get_cbtool_data_old(experiment_path: str, instances_n: int):
    csv_path = glob.glob(f'{experiment_path}/VM_runtime_app_*.csv')[0]
    df = pd.read_csv(csv_path, sep=',', skiprows=28)
    df = df.get(['time_h', 'ai_name', 'app_throughput', 'app_latency'])
    df['time_h'] = df['time_h'].apply(convert_from_time_str)
    df = df.rename(columns={'time_h': 'time'})
    df = add_instances_n_old(df, instances_n=instances_n)
    df = df[df['ai_name'] == 'ai_1']
    return df


def add_instances_n(df: pd.DataFrame, instances_n: int):
    start_times = []
    end_times = []
    
    for i in range(1, instances_n + 1):
        ai_name = f'ai_{i}'
        first_entry = df[df['ai_name'] == ai_name].iloc[0]
        start_times.append(first_entry['time'])
        end_times.append(first_entry['time'] - first_entry['app_completion_time'])
    end_times.append(df.iloc[-1]['time'] + 1)
    
    for i, (start_time, end_time) in enumerate(zip(start_times, end_times[1:]), 1):
        df.loc[(start_time <= df['time']) & (df['time'] <= end_time), 'instances_n'] = i

    return df.dropna(axis=0, subset=['instances_n'])


def get_cbtool_data(experiment_path: str, instances_n: int):
    csv_path = glob.glob(f'{experiment_path}/VM_runtime_app_*.csv')[0]
    df = pd.read_csv(csv_path, sep=',', skiprows=28)
    df = df.get(['time_h', 'ai_name', 'app_throughput', 'app_latency', 'app_completion_time'])
    df['time_h'] = df['time_h'].apply(convert_from_time_str)
    df = df.rename(columns={'time_h': 'time'})
    df = add_instances_n(df, instances_n=instances_n)
    df = df[df['ai_name'] == 'ai_1']
    df['app_throughput_inv'] = 1. / df['app_throughput']
    df['instances_n'] = df['instances_n'].astype(int)
    return df


def get_cpu_data(experiment_path: str):
    cpu_csv_path = glob.glob(f'{experiment_path}/resources/metric_node_*_cpu.csv')[0]
    df_cpu = pd.read_csv(cpu_csv_path)
    df_cpu = df_cpu.get(['time', 'value'])
    df_cpu['time'] = df_cpu['time'].apply(convert_from_padded)
    df_cpu = df_cpu.rename(columns={'value': 'cpu'})
    df_cpu['cpu'] = df_cpu['cpu'].astype(float)
    return df_cpu


def get_mem_data(experiment_path: str):
    mem_csv_path = glob.glob(f'{experiment_path}/resources/metric_node_*_memory.csv')[0]
    df_mem = pd.read_csv(mem_csv_path)
    df_mem = df_mem.get(['time', 'value'])
    return df_mem


def clean_column_names(df: pd.DataFrame):
    df.columns = [re.sub('[".,]', '', re.sub('[{}()=\-\+\s#~:/]', '_', column)) for column in df.columns]
    return df


def remove_nan_columns(df: pd.DataFrame):
    return df[df.columns[~df.isnull().any()]]


def get_os_metrics(experiment_path: str):
    os_metrics_path = f'{experiment_path}/resources/metric_os.csv'
    df_os = pd.read_csv(os_metrics_path)
    df_os['time'] = df_os['time'].apply(convert_from_padded)
    df_os = df_os.drop('name', axis=1)
    df_os = clean_column_names(df_os)
    return remove_nan_columns(df_os)


def get_closest_row(df: pd.DataFrame, time: int):
    lower_idx = bisect_left(df['time'].values, time)
    higher_idx = bisect_right(df['time'].values, time)
    if higher_idx == lower_idx:  # val is not in the list
        closest_idx = lower_idx - 1 if abs(df['time'][lower_idx - 1] - time) < abs(df['time'][lower_idx] - time) else lower_idx
        return df.iloc[closest_idx]
    else:  # exact match
        return df.iloc[lower_idx]


def merge_dataframes_old(df_cbt: pd.DataFrame, df_cpu: pd.DataFrame, df_mem: pd.DataFrame, df_os: pd.DataFrame, max_time_diff=5):
    # Cpu and memory correspond to each other (same timestamps), so we can simply put them to the same df
    df_cpu['memory'] = df_mem['value']
    
    res = []    
    for i, cbt_row in df_cbt.iterrows():
        cbt_time = cbt_row['time']
        closest_cpu_row = get_closest_row(df_cpu, cbt_time)
        
        if abs(closest_cpu_row['time'] - cbt_time) > max_time_diff:
            continue  # ts diff too high - we skip that row
            
        closest_os_row = get_closest_row(df_os, cbt_time)
        
        if abs(closest_os_row['time'] - cbt_time) > max_time_diff:
            continue  # ts diff too high - we skip that row

        res.append([
            str(cbt_time),
            str(int(closest_cpu_row['time'])),
            cbt_row['app_latency'],
            cbt_row['app_throughput'],
            closest_cpu_row['cpu'],
            closest_cpu_row['memory'],
            cbt_row['instances_n'],
            str(int(closest_os_row[0])),
        ] + list(closest_os_row[1:]))
    return pd.DataFrame(res, columns=(['cbtool_time', 'cpu_time', 'app_latency', 'app_throughput', 'cpu', 'memory', 'instances_n', 'os_time'] + list(df_os.columns[1:])))


def get_fitting_cpu_mean(cbt_row: pd.Series, df_cpu: pd.DataFrame, cpu_window: int):
    end_time = cbt_row['time']
    start_time = end_time - cbt_row['app_completion_time']
    cpu_between = df_cpu[(start_time <= df_cpu['time'] - cpu_window) & (df_cpu['time'] <= end_time)]
    
    rows = list(cpu_between.iterrows())
    to_remove = []
    i = 0
    j = 1
    while j < len(rows):
        idx_i, row_i = rows[i]
        idx_j, row_j = rows[j]
        
        if row_i['time'] > row_j['time'] - cpu_window:
            to_remove.append(idx_j)
        else:
            i = j
        j += 1
    
    cpu_mean = cpu_between.drop(to_remove).mean()
    return cpu_mean['cpu'], cpu_mean['memory']


def get_os_metrics_mean(cbt_row: pd.Series, df_os: pd.DataFrame):
    end_time = cbt_row['time']
    start_time = end_time - cbt_row['app_completion_time']
    os_between = df_os[(start_time <= df_os['time']) & (df_os['time'] <= end_time)]
    return os_between.mean()


def merge_dataframes(df_cbt: pd.DataFrame, df_cpu: pd.DataFrame, df_mem: pd.DataFrame, df_os: pd.DataFrame, cpu_window: int):
    df_cpu['memory'] = df_mem['value']
    
    res = []
    for i, cbt_row in df_cbt.iterrows():
        cbt_time = cbt_row['time']
        cpu, mem = get_fitting_cpu_mean(cbt_row, df_cpu, cpu_window=cpu_window)
        
        if np.isnan(cpu):
            continue
        
        os_row = get_os_metrics_mean(cbt_row, df_os)

        res.append([
            str(cbt_time),
            cbt_row['app_latency'],
            cbt_row['app_throughput'],
            cbt_row['app_throughput_inv'],
            cpu,
            mem,
            cbt_row['instances_n'],
            cbt_row['app_completion_time']
        ] + list(os_row[1:]))
    return pd.DataFrame(res, columns=(['cbtool_time', 'app_latency', 'app_throughput', 'app_throughput_inv', 'cpu', 'memory', 'instances_n', 'app_completion_time'] + list(df_os.columns[1:])))


def merge_dataframes_cpu(df_cbt: pd.DataFrame, df_cpu: pd.DataFrame, df_mem: pd.DataFrame, cpu_window: int):
    df_cpu['memory'] = df_mem['value']
    
    res = []
    for i, cbt_row in df_cbt.iterrows():
        cbt_time = cbt_row['time']
        cpu, mem = get_fitting_cpu_mean(cbt_row, df_cpu, cpu_window=cpu_window)
        
        if np.isnan(cpu):
            continue

        res.append([
            str(cbt_time),
            cbt_row['app_latency'],
            cbt_row['app_throughput'],
            cbt_row['app_throughput_inv'],
            cpu,
            mem,
            cbt_row['instances_n'],
            cbt_row['app_completion_time'],
        ])
    return pd.DataFrame(res, columns=(['cbtool_time', 'app_latency', 'app_throughput', 'app_throughput_inv', 'cpu', 'memory', 'instances_n', 'app_completion_time']))


def get_data_with_metrics_old(experiment_path: str, instances_n: int, max_time_diff=5):
    df_cbt = get_cbtool_data_old(experiment_path, instances_n=instances_n)
    df_cpu = get_cpu_data(experiment_path)
    df_mem = get_mem_data(experiment_path)
    df_os = get_os_metrics(experiment_path)
    return merge_dataframes_old(df_cbt, df_cpu, df_mem, df_os, max_time_diff=max_time_diff)


def get_data_with_metrics(experiment_path: str, instances_n: int, cpu_window=30):
    df_cbt = get_cbtool_data(experiment_path, instances_n=instances_n)
    df_cpu = get_cpu_data(experiment_path)
    df_mem = get_mem_data(experiment_path)
    df_os = get_os_metrics(experiment_path)
    return merge_dataframes(df_cbt, df_cpu, df_mem, df_os, cpu_window=cpu_window)


def get_data_with_cpu(experiment_path: str, instances_n: int, cpu_window=30):
    df_cbt = get_cbtool_data(experiment_path, instances_n=instances_n)
    df_cpu = get_cpu_data(experiment_path)
    df_mem = get_mem_data(experiment_path)
    return merge_dataframes_cpu(df_cbt, df_cpu, df_mem, cpu_window=cpu_window)


def trim_experiment(df: pd. DataFrame, max_instances_n: int):
    return df[df['instances_n'] <= max_instances_n]



def extract_datetime_from_log(line: str, year=2020):
    """Take one log line from CBTOOL log and extract the datetime.
    
    Function supports logs from root_operations.log.
    It is neceessary to provide the year, as the logs do not contain it.
    """
    month, day, t = line.split()[:3]
    return datetime.strptime(f'{year} {month} {day} {t}', '%Y %b %d %H:%M:%S')


def get_setup_intervals(path: str, year=2020):
    """Parse the logs and get datetime intervals for setting up phase of each load."""
    LEFT_END_PHRASE = 'AI object initialization success'
    RIGHT_END_PHRASE = 'It is ssh-accessible at the IP address'
    left_ends = []
    right_ends = []
    
    with open(path + '/root_operations.log') as file:
        for line in file:
            if LEFT_END_PHRASE in line:
                left_ends.append(extract_datetime_from_log(line, year=year))
            if RIGHT_END_PHRASE in line:
                right_ends.append(extract_datetime_from_log(line, year=year))
    return list(zip(left_ends[1:], right_ends[2:]))


def remove_setup_datapoints(df: pd.DataFrame, path: str, year=2020):
    """Remove datapoints that are influenced by setup phase of deployed workloads."""
    completion_time = int(df['app_completion_time'].max())
    
    for left, right in get_setup_intervals(path, year=year):
        df = df[(df['time'] < left) | (right + timedelta(seconds=completion_time) < df['time'])]
    return df
