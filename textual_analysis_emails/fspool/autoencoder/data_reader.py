import pandas as pd
import random
import os
import numpy as np
import cardinality
from tqdm import tqdm_notebook
import torch

def pad_zeros(points, cardinality, _type='torch', pad_value = 0):
    if _type=='np':
        if points.shape[2] < cardinality:
            # pad to fixed size
            if pad_value == 0:
                padding = np.zeros((points.shape[0], points.shape[1], cardinality - points.shape[2]), dtype=float)
            else:
                padding = np.ones((points.shape[0], points.shape[1], cardinality - points.shape[2]), dtype=float)*pad_value
            points = np.concatenate([points, padding], axis=2)
    else:
        if points.size(2) < cardinality:
            # pad to fixed size
            if pad_value == 0:
                padding = torch.zeros(points.size(0), points.size(1), cardinality - points.size(2)).to(points.device)
            else:
                padding = (torch.ones(points.size(0), points.size(1), cardinality - points.size(2))*pad_value).to(points.device)
            points = torch.cat([points, padding], dim=2)
    return points

def load_bipartite_hypergraph(data_params):
    id_p_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_label_file']), sep='\t', header=None)
    id_a_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['u_label_file']), sep='\t', header=None)
    id_a_map = dict(zip(id_a_map[0], id_a_map[1]))
    id_k_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['v_label_file']), sep='\t', header=None)
    id_k_map = dict(zip(id_k_map[0], id_k_map[1]))
    p_a_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_u_list_file']), sep=':', header=None)
    p_k_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_v_list_file']), sep=':', header=None)
    n_p, na, nk = len(id_p_map), len(id_a_map), len(id_k_map)
    U = list(map(lambda x: list(map(int, x.split(','))), p_a_list_map[1]))
    V = list(map(lambda x: list(map(int, x.split(','))), p_k_list_map[1]))
    return U, V

def load_bipartite_hypergraph_with_vector(data_params, mask_flag = True, neg_pos_ratio = 1):
    embeddings = load(open(os.path.join(data_params['home_path'], data_params['emb_pkl_file']), 'rb'))
    id_p_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_label_file']), sep='\t', header=None)
    id_a_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['u_label_file']), sep='\t', header=None)
    id_a_map = dict(zip(id_a_map[0], id_a_map[1]))
    id_k_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['v_label_file']), sep='\t', header=None)
    id_k_map = dict(zip(id_k_map[0], id_k_map[1]))
    p_a_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_u_list_file']), sep=':', header=None)
    p_k_list_map = pd.read_csv(os.path.join(data_params['home_path'], data_params['r_v_list_file']), sep=':', header=None)
    n_p, na, nk = len(id_p_map), len(id_a_map), len(id_k_map)
    U = list(map(lambda x: list(map(int, x.split(','))), p_a_list_map[1]))
    V = list(map(lambda x: list(map(int, x.split(','))), p_k_list_map[1]))
    neg_U, neg_V = get_neg_samp(U, V, neg_pos_ratio)
    labels = np.array([1]*len(U) + [0]*len(neg_U))
    U = U + list(neg_U)
    V = V + list(neg_V)
    ax_map = {a: embeddings[str(n_p+nk+a)] for a in id_a_map}
    kx_map = {k: embeddings[str(n_p+k)] for k in id_k_map}
    U = list(map(lambda x: np.array(list(map(ax_map.get, x))).T, U))
    V = list(map(lambda x: np.array(list(map(kx_map.get, x))).T, V))
    n_points_U = np.array([x.shape[1] for x in U])
    n_points_V = np.array([x.shape[1] for x in V])
    cardinality_U = max(n_points_U)
    cardinality_V = max(n_points_V)
    U = np.concatenate([pad_zeros(np.array([x]), cardinality_U, 'np') for x in U])
    V = np.concatenate([pad_zeros(np.array([x]), cardinality_V, 'np') for x in V])
    if mask_flag:
        mask = np.array([[1]*n_points_U[i] + [0]*(U.shape[-1]-n_points_U[i]) for i in range(U.shape[0])])
        U = np.concatenate((U, np.broadcast_to(mask[:, None, :], (U.shape[0], 1, cardinality_U))), axis = -2)
        mask = np.array([[1]*n_points_V[i] + [0]*(V.shape[-1]-n_points_V[i]) for i in range(V.shape[0])])
        V = np.concatenate((V, np.broadcast_to(mask[:, None, :], (V.shape[0], 1, cardinality_V))), axis = -2)
    return U, V, n_points_U, n_points_V, cardinality_U, cardinality_V, labels
