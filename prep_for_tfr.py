'''
This file converts data of groups of features (in this case, impressions) into document-query libSVM file format
Original file written by Joseph Palermo
'''

import os
import numpy as np
import pandas as pd
from utils import get_logger, get_data_path

Filepath = get_data_path()

pd.options.display.max_columns = 250

def has_num(string):
    return any(ch.isdigit() for ch in string)

def get_string_to_sort_by(col_name):
    return col_name.split('_')[-1].zfill(2) + "_".join(col_name.split('_')[:-1])

def qid_array_to_libsvm_lines(qid_arr):
    lines = []
    for i in range(len(qid_arr)):
        query_document_arr = qid_arr[i,:]
        line = get_line(query_document_arr)
        lines.append(line)
    return "\n".join(lines)

def get_line(query_document_arr):
    target = query_document_arr[0]
    values = query_document_arr[1:]
    value_strings = [str(int(target))]
    col_names = ['qid']
    col_names.extend([str(i) for i in range(1,len(values))])
    values = [str(int(value)) if col_name in int_cols else value for col_name, value in zip(col_names, values)]
    value_strings.extend([f"{col_name}:{value}" for col_name, value in zip(col_names, values)])
    return " ".join(value_strings)

# load data
mode = 'TRAIN'
if mode == 'TRAIN':
    pq_path = 'tfr_train.parquet'
elif mode == 'VALID':
    pq_path = 'tfr_valid.parquet'
else:
    pq_path = 'tfr_train.parquet'

train_data_path = os.path.join(Filepath.gbm_cache_path, pq_path)
#small_train_data_path = 'data/train_small.snappy'
df = pd.read_parquet(train_data_path)
df.shape

n_imps_reference = df['n_imps_original']
df.drop(columns=['n_imps_original'], inplace=True)

# drop cols
#impressions_cols = [f'impressions_{i}' for i in range(25)]
current_filter_cols = [f'current_filters_{i}' for i in range(33)]
#cols_to_drop = ['step', 'session_id', 'reference', 'latest_clickout'] + impressions_cols + current_filter_cols
cols_to_drop = current_filter_cols
df = df.drop(cols_to_drop, axis=1)

df['qid'] = df.index
df.shape

#rename impressions_0_priors --> impressions_0 for the sort. ugh.
df.rename(columns=dict(zip([c for c in df.columns if 'priors' in c], [c.replace('_priors', '') for c in  df.columns if 'priors' in c])),inplace=True)

# re-order cols
n_cols = len(df.columns)
prefix_cols = ['target', 'qid']
non_num_cols = [col for col in df.columns if not has_num(col) and col not in prefix_cols]
remaining_cols = list(set(df.columns) - set(prefix_cols + non_num_cols))
reamining_cols_order = np.argsort([get_string_to_sort_by(col) for col in remaining_cols])
remaining_cols_arr = np.array(remaining_cols)
remaining_cols = remaining_cols_arr[reamining_cols_order]
start_cols = prefix_cols + non_num_cols
col_order = start_cols + list(remaining_cols)
df = df[col_order]


remaining_cols = df.columns[6::] # select only the groups of 25
start_cols = list(df.columns[0:6])

# constants
n_documents = 25
feature_group_size = len(remaining_cols) // n_documents
#n_train = int(len(df) * 0.9)

# column lists
query_document_col_names = start_cols + list(['_'.join(col.split('_')[:-1]) for col in remaining_cols[:feature_group_size]])
int_cols = set(['qid','session_size','last_click','last_interact','prev_click','prev_interact'])

document_df = df[remaining_cols]
with open(os.path.join(Filepath.gbm_cache_path, '{}_normalized.text'.format(mode)), 'w') as f:
    for i in list(range(len(df))):  #[:n_train]:
        if i % 10000 == 0:
            print(i)
        n_imps = n_imps_reference.iloc[i]
        i_document_df = document_df.iloc[i:i + 1]
        a = i_document_df.values
        col_cut_off_index = int(feature_group_size * n_imps)
        a = a[:, :col_cut_off_index]
        documents = np.reshape(a, (-1, feature_group_size))
        i_start_df = df.iloc[i:i + 1][start_cols]
        prefix_arr = i_start_df.values
        target = int(prefix_arr[0, 0])
        prefix_arr = np.repeat(prefix_arr, [n_imps], axis=0)
        prefix_arr[:, 0] = 0
        prefix_arr[target, 0] = 1
        qid_arr = np.concatenate([prefix_arr, documents], axis=1)
        lines = qid_array_to_libsvm_lines(qid_arr)
        f.write(lines + '\n')

'''
with open('data/validation.text', 'w') as f:
    for i in list(range(len(df)))[n_train:]:
        if i % 10000 == 0:
            print(i)
        n_imps = df.iloc[i]['n_imps']
        i_document_df = document_df.iloc[i:i + 1]
        a = i_document_df.values
        col_cut_off_index = int(feature_group_size * n_imps)
        a = a[:, :col_cut_off_index]
        documents = np.reshape(a, (-1, feature_group_size))
        i_start_df = df.iloc[i:i + 1][start_cols]
        prefix_arr = i_start_df.values
        target = int(prefix_arr[0, 0])
        prefix_arr = np.repeat(prefix_arr, [n_imps], axis=0)
        prefix_arr[:, 0] = 0
        prefix_arr[target, 0] = 1
        qid_arr = np.concatenate([prefix_arr, documents], axis=1)
        lines = qid_array_to_libsvm_lines(qid_arr)
        f.write(lines + '\n')
'''