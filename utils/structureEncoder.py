import torch
import numpy as np
from HPGT.utils.tools import calculate_degree_mtrx
import math


def DB_random_walk(edge_index, start_node, walk_length, num_walks, deg_condition_prob, deg):
    walks = []
    for i in range(num_walks):
        walk_path = [start_node]
        while len(walk_path) < walk_length:
            cur = walk_path[-1]
            cur_nbrs = edge_index[1, edge_index[0] == cur]
            trans_cond_prob = deg_condition_prob[deg[cur].long(), deg[cur_nbrs].long()]
            trans_prob = (trans_cond_prob / (sum(trans_cond_prob)) + 1e-30).detach().numpy()
            cur_nbrs_list = list(cur_nbrs.detach().numpy())
            if len(cur_nbrs_list) > 0:
                walk_path.append(np.random.choice(cur_nbrs_list, p=trans_prob.ravel()))
            else:
                break
        walks.append(walk_path)
        walk_path = []
    return walks  # （num_walks,walk_length）


def random_walk(edge_index, start_node, walk_length, num_walks):
    walks = []
    for i in range(num_walks):
        walk_path = [start_node]
        while len(walk_path) < walk_length + 1:
            cur = walk_path[-1]
            cur_nbrs = edge_index[1, edge_index[0] == cur]
            cur_nbrs_list = list(cur_nbrs.detach().numpy())
            if len(cur_nbrs_list) > 0:
                walk_path.append(np.random.choice(cur_nbrs_list))
            else:
                break
        del (walk_path[0])
        walks.append(walk_path)
        walk_path = []
    return walks


def get_x_based_walks(x, edge_index, walk_length, num_walks, deg, type):
    if type == 'DBP':
        deg_trans_prob = calculate_degree_mtrx(edge_index, deg)
        walks_list = []

        for i in range(x.shape[0]):
            walks = DB_random_walk(edge_index, i, walk_length, num_walks, deg_trans_prob, deg)
            walks_list.append(walks)
        walks_2_tensor = torch.tensor(walks_list)
        walks_2_tensor = walks_2_tensor.flatten()
        x_based_walks = x[walks_2_tensor]
        return x_based_walks.reshape(x.shape[0], num_walks, walk_length, x.shape[1])

    if type == 'RWP':
        walks_list = []
        for i in range(x.shape[0]):
            walks = random_walk(edge_index, i, walk_length, num_walks)
            walks_list.append(walks)
        walks_2_tensor = torch.tensor(walks_list)
        walks_2_tensor = walks_2_tensor.flatten()
        x_based_walks = x[walks_2_tensor]
        return x_based_walks.reshape(x.shape[0], num_walks, walk_length, x.shape[1])


def get_se(x, edge_index, walk_length, num_walks, deg, type):
    if type == 'DBP':
        deg_trans_prob = calculate_degree_mtrx(edge_index, deg)
        walks_list = []

        for i in range(x.shape[0]):
            walks = DB_random_walk(edge_index, i, walk_length, num_walks, deg_trans_prob, deg)
            walks_list.append(walks)
        walks_2_tensor = torch.tensor(walks_list)
        walks_2_tensor = walks_2_tensor.reshape(x.shape[0], -1)

        se = torch.zeros_like(x)
        for i in range(x.shape[0]):
            del_se_idx = walks_2_tensor[i]
            save_se_idx = del_se_idx[~np.isin(del_se_idx, del_se_idx[0])]
            if len(save_se_idx) == 0:
                save_se_idx = del_se_idx[0]
            save_se = x[save_se_idx].reshape(-1, x.shape[1])
            sum_se = save_se.sum(0)
            se[i] = se[i] + sum_se
        return se
