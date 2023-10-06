import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch_geometric.utils import degree, add_self_loops, to_scipy_sparse_matrix, is_undirected, to_undirected
from HPGT.utils.Classifier_layer import DBPGCN, DBPMLP, DBPGAT, DBPGIN, DBPSAGE
import numpy as np
from sklearn import metrics


plt.rcParams["font.sans-serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.size'] = 30
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams["font.family"] = 'Times New Roman'



def visualize_graph(G, color):
    '''
    :param G: G=to_networkx(data,to_undirected=True)
    :param color: data.y labels
    :return:None
    '''
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(outputs, labels):
    outputs = outputs.detach().numpy()
    labels = labels.detach().numpy()

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(outputs)

    color_idx = {}
    for i in range(outputs.shape[0]):
        label = labels[i]
        color_idx.setdefault(label.item(), [])
        color_idx[label.item()].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


def nmi(a, b):
    result_NMI = metrics.normalized_mutual_info_score(a, b)
    print("result_NMI:", result_NMI)


def draw_degree_correlation_mtx(name, data):
    deg = degree(data.edge_index[0])
    deg = deg.int()
    deg_unique = torch.unique(deg).detach().numpy()
    max_deg = int(deg.max().detach().numpy())
    min_deg = int(deg.min().detach().numpy())
    deg_cor_matrix = torch.zeros((max_deg - min_deg + 1, max_deg - min_deg + 1))
    edge_index = data.edge_index.detach().numpy()
    num_edge = data.num_edges
    for i in range(num_edge):
        node_s_deg = deg[edge_index[0, i]].long()
        node_t_deg = deg[edge_index[1, i]].long()
        deg_cor_matrix[node_s_deg - 1, node_t_deg - 1] += 1
    sum_row = torch.sum(deg_cor_matrix, dim=1)
    save_col = (sum_row != 0)
    save_row = save_col.t()
    deg_cor_matrix = deg_cor_matrix[:, save_col]
    deg_cor_matrix = deg_cor_matrix[save_row, :]
    joint_probability = deg_cor_matrix / data.num_edges

    dict_ = {'label': 'Degree correlation'}
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(joint_probability.detach().numpy(), columns=deg_unique,
                             index=deg_unique), annot=False, vmax=joint_probability.max(), vmin=0, xticklabels=True,
                yticklabels=True, square=True, cmap="YlGnBu", cbar_kws=dict_)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    ax.set_ylabel('Degree', fontsize=30)
    ax.set_xlabel('Degree', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.xlabel('度',fontsize=20)
    # plt.ylabel('度',fontsize=20)
    plt.savefig('./figs/' + name + '_DRM.svg', bbox_inches='tight')


def edge_index_to_adj(edge_index, edge_values=None, add_self=False):
    if add_self is True:
        edge_index = add_self_loops(edge_index=edge_index)
    adj_csc = to_scipy_sparse_matrix(edge_index=edge_index, edge_attr=edge_values).tocsc()
    adj = torch.tensor(adj_csc.toarray())
    return adj


def edge_index_to_matrix(x, edge_index, edge_values=None, add_self=False):
    if add_self is True:
        edge_index = add_self_loops(edge_index=edge_index)
    matrix_shape = np.zeros((x.shape[0], x.shape[0])).shape
    if edge_values is None:
        edge_values = torch.FloatTensor(np.ones(edge_index.shape[1]))
    matrix = torch.sparse_coo_tensor(edge_index, edge_values, matrix_shape)
    return matrix


def node_transfer_mtx(data):
    data.edge_index = add_self_loops(edge_index=data.edge_index)[0]

    deg = degree(data.edge_index[0])
    A = edge_index_to_adj(edge_index=data.edge_index)

    deg = deg.int()
    deg_unique = torch.unique(deg).detach().numpy()

    D = torch.zeros((data.x.shape[0], len(deg_unique)))
    K = torch.ones((data.x.shape[0], len(deg_unique)))

    dict_deg = {}
    for i in range(len(deg_unique)):
        dict_deg[deg_unique[i]] = i

    deg_list = deg.detach().numpy()
    # remark_deg
    for i in range(len(deg)):
        idx = dict_deg[deg_list[i]]
        D[i, idx] = 1

    DTA = torch.mm(D.t(), A)
    DTAD = torch.mm(DTA, D)
    DTAK = torch.mm(DTA, K)
    B = DTAD / DTAK
    return B


def remove_edge(edge_index, u, v):
    del_src_u_mask = (edge_index[0] == u)
    del_dst_v_mask = (edge_index[1] == v)
    del_edge_uv_mask = del_dst_v_mask * del_src_u_mask
    del_src_v_mask = (edge_index[0] == v)
    del_dst_u_mask = (edge_index[1] == u)
    del_edge_vu_mask = del_dst_u_mask * del_src_v_mask
    del_edge_mask = del_edge_vu_mask + del_edge_uv_mask
    save_edge_mask = abs(del_edge_mask.int() - 1).bool()
    adjust_edge_index = edge_index[save_edge_mask]
    return adjust_edge_index


def adj_to_coo(adj):
    idx = torch.nonzero(adj).T
    data = adj[idx[0], idx[1]]
    adj_coo = torch.sparse_coo_tensor(idx, data, adj.shape)
    edge_index = adj_coo._indices()
    edge_weight = adj_coo._values()
    return edge_index, edge_weight


def view_parameters(model, param_name=None):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()
    if param_name is None:
        return params
    else:
        return params[param_name]


def draw_loss(Loss_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = Loss_list
    plt.title(Type_name + ' loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel(Type_name + ' loss', fontsize=20)
    plt.grid()
    plt.savefig('./figs/' + dataset + '/' + Type_name + '_loss.png')
    plt.show()


def draw_acc(acc_list, epochs, dataset, Type_name='Train'):
    plt.cla()
    x1 = range(1, epochs + 1)
    y1 = acc_list
    plt.title(Type_name + ' accuracy vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(Type_name + ' accuracy', fontsize=20)
    plt.grid()
    plt.savefig('./figs/' + dataset + '/' + Type_name + '_accuracy.png')
    plt.show()


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def calculate_degree_mtrx(edge_index, deg):
    max_deg = int(deg.max().detach().numpy())
    deg_cor_matrix = torch.zeros((max_deg + 1, max_deg + 1))
    edge_index = edge_index.detach().numpy()
    num_edge = edge_index.shape[1]
    for i in range(num_edge):
        node_s_deg = deg[edge_index[0, i]].long()
        node_t_deg = deg[edge_index[1, i]].long()
        deg_cor_matrix[node_s_deg, node_t_deg] += 1

    conditional_probability = deg_cor_matrix / (
            (torch.sum(deg_cor_matrix, 1).unsqueeze(-1)) + 1e-30)
    return conditional_probability


def train_val_test_split(data, train_p, val_p):
    node_index = list(range(0, data.num_nodes))
    np.random.shuffle(node_index)

    train_size, val_size = int(data.num_nodes * train_p), int(data.num_nodes * val_p)

    train_idx = node_index[0:train_size]
    val_idx = node_index[train_size:train_size + val_size]
    test_idx = node_index[train_size + val_size:]

    train_mask = torch.zeros(data.num_nodes)
    val_mask = torch.zeros(data.num_nodes)
    test_mask = torch.zeros(data.num_nodes)

    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1
    return train_mask.bool(), val_mask.bool(), test_mask.bool()


def set_batch(batch_size, x, y, mask):
    index = torch.tensor(range(0, len(mask)))[mask]
    batch_size = batch_size
    mod = x.shape[0] % batch_size
    pad_num = batch_size - mod
    pad_x = x[0:pad_num]
    pad_y = y[0:pad_num]
    pad_index = index[0:pad_num]
    batch_x = torch.cat((x, pad_x), dim=0).reshape(-1, batch_size, x.shape[1])
    batch_y = torch.cat((y, pad_y), dim=-1).reshape(-1, batch_size)
    batch_index = torch.cat((index, pad_index), dim=-1).reshape(-1, batch_size)
    return batch_x, batch_y, batch_index


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.test_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, test_acc, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.test_acc = test_acc
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:  # val_loss increase or stable
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.test_acc = test_acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def select_mask(i: int, train: torch.Tensor, val: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    if train.dim() == 1:
        return train, val, test
    else:
        indices = torch.tensor([i]).to(train.device)
        train_idx = torch.index_select(train, 1, indices).reshape(-1)
        val_idx = torch.index_select(val, 1, indices).reshape(-1)
        test_idx = torch.index_select(test, 1, indices).reshape(-1)
        return train_idx, val_idx, test_idx


def load_model(model_type, num_nodes, in_dim, hidden_dim, out_dim, num_walks,
               walks_length, walks_heads, num_layers, transformer_heads,
               dropout, aggr, use_se):
    if model_type == 'DBPGCN':
        return DBPGCN(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                      num_walks=num_walks, walks_length=walks_length, walks_heads=walks_heads,
                      num_layers=num_layers, transformer_heads=transformer_heads
                      , dropout=dropout, aggr=aggr, use_se=use_se)
    elif model_type == 'DBPGAT':
        return DBPGAT(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                      num_walks=num_walks, walks_length=walks_length, walks_heads=walks_heads,
                      num_layers=num_layers, transformer_heads=transformer_heads
                      , dropout=dropout, aggr=aggr, use_se=use_se)
    elif model_type == 'DBPGIN':
        return DBPGIN(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                      num_walks=num_walks, walks_length=walks_length, walks_heads=walks_heads,
                      num_layers=num_layers, transformer_heads=transformer_heads
                      , dropout=dropout, aggr=aggr, use_se=use_se)
    elif model_type == 'DBPSAGE':
        return DBPSAGE(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                       num_walks=num_walks, walks_length=walks_length, walks_heads=walks_heads,
                       num_layers=num_layers, transformer_heads=transformer_heads
                       , dropout=dropout, aggr=aggr, use_se=use_se)
    else:
        return DBPMLP(num_nodes=num_nodes, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                      num_walks=num_walks, walks_length=walks_length, walks_heads=walks_heads,
                      num_layers=num_layers, transformer_heads=transformer_heads
                      , dropout=dropout, aggr=aggr, use_se=use_se)
