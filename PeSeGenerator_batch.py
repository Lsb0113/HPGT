import torch

from HPGT.utils.positionEncoder import choose_pe
from HPGT.utils.structureEncoder import get_x_based_walks, get_se
from HPGT.utils.data import get_dataset
from torch_geometric.utils import degree, add_self_loops, remove_self_loops, is_undirected, to_undirected
from torch_geometric.loader import ClusterLoader, ClusterData
from HPGT.utils.tools import set_random_seed


def generate_se(data, num_walks, walks_length, use_se, Type):
    # Type includes utils and RWP
    x_walks = None

    if use_se:
        deg = degree(data.edge_index[0])
        if is_undirected(edge_index=data.edge_index) is False:
            print('is directed')
            data.edge_index = to_undirected(edge_index=data.edge_index)
        if sum(deg == 0) == 0:
            edge_index = data.edge_index
            deg = degree(edge_index[0])
        else:
            edge_index = add_self_loops(data.edge_index)[0]
            deg = degree(edge_index[0])
        x_walks = get_x_based_walks(x=data.x, edge_index=edge_index, walk_length=walks_length,
                                    num_walks=num_walks, deg=deg, type=Type)
    return x_walks


def get_batch_se(dataset_name, data, num_walks, walks_length, use_se, Type):
    x_walks = generate_se(data=data, num_walks=num_walks,
                          walks_length=walks_length,
                          use_se=use_se, Type=Type)

    if use_se:
        torch.save(x_walks,
                   'walks/' + dataset_name + 'Walks/' + Type + '_' + dataset_name + '_walks' + str(
                       walks_length) + '.pt')
