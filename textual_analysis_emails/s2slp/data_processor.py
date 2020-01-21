import pandas as pd
import random
import numpy as np
import cardinality
from tqdm import tqdm_notebook
import networkx as nx
from sklearn.model_selection import train_test_split

def get_neg_samp(U, V, num_neg):
    edges = set(zip(map(frozenset, U), map(frozenset, V)))
    setU = set(map(frozenset, U))
    setV = set(map(frozenset, V))
    non_edges = set()
    num_pos = len(U)
    num_total = len(setU)*len(setV)
    max_num_neg = num_total-num_pos
    if num_neg > max_num_neg:
        print('WARNING: Too many negative samples demanded.')
        num_neg = max_num_neg
    while len(non_edges) < num_neg:
        u = random.sample(setU, 1)[0]
        v = random.sample(setV, 1)[0]
        pair = (u, v)
        if pair in edges or pair in non_edges:
            continue
        non_edges.add(pair)
    neg_U, neg_V = zip(*map(lambda x: list(map(list, x)), non_edges))
    assert not all([x in non_edges for x in set(zip(map(frozenset, U), map(frozenset, V)))])
    assert not all([x in edges for x in set(zip(map(frozenset, neg_U), map(frozenset, neg_V)))])
    return neg_U, neg_V

def data_process(pos_A, pos_B, neg_pos_ratio = 1, unobs_ratio = 0.2):
    """WARNING: This function offsets set V's node ids"""
    j_ = [max(x) for x in list(pos_A)]
    V_offset=max(j_) + 1

    b_=[]
    for j in pos_B:
        b_.append([(x + V_offset) for x in j])
    j_=[max(x) for x in b_]
    key_num=max(j_) + 1
    r=range(key_num + 1)

    # key_num
    # data = total_data
    pos_data=list(zip(list(pos_A),(b_)))
    obs_pos, unobs_pos = train_test_split(pos_data, test_size = unobs_ratio)

    G_obs = Graph_from_data(r, obs_pos)
    num_neg = len(unobs_pos)*neg_pos_ratio
    neg_A, neg_B = get_neg_samp(pos_A, b_, num_neg)
    neg_data = list(zip(list(neg_A),(neg_B)))

    lab=[1]*len(unobs_pos)+[0]*(len(neg_data))
    test_ = unobs_pos + neg_data
    unobs_data = list(zip(test_, lab))

    train_data, test_data = train_test_split(unobs_data, test_size=0.2)
    # pos_A = [{1, 2}, {4, 2}, {4, 5, 3}]
    # pos_B = [{4}, {2, 3, 1, 5}, {3, 2}]
    # G, obs_pos, train_data, test_data, V_offset = data_process(pos_A, pos_B, neg_pos_ratio = 1, unobs_ratio=0.5)

    return G_obs, obs_pos, train_data, test_data, V_offset


def Graph_from_data(nodes_list,train_data):
    G=nx.Graph()
    G.add_nodes_from(nodes_list)
    for i in range(len(train_data)):
        for k in train_data[i][0]:
            for m in train_data[i][1]:
                G.add_edge(k,m,weight=1)
    return G

