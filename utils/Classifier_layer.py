import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv

from HPGT.utils.model import GraphTransformer


class DBPnet(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True, **kwargs):
        super(DBPnet, self).__init__(**kwargs)

        self.dropout = dropout

        self.GT = GraphTransformer(num_nodes=num_nodes, in_dim=in_dim, out_dim=out_dim, num_walks=num_walks,
                                   walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                                   transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)

    def forward(self, data, walks, deg):
        x, edge_index = data.x, data.edge_index
        gt_output = self.GT(x, walks, deg)
        gt_output = F.relu(gt_output)
        gt_output = F.dropout(gt_output, p=self.dropout, training=self.training)
        return gt_output


class DBPMLP(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True):
        super(DBPMLP, self).__init__()

        self.dropout = dropout

        self.net = DBPnet(num_nodes=num_nodes, in_dim=in_dim, out_dim=hidden_dim, num_walks=num_walks,
                          walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                          transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)
        self.lin1 = nn.Linear(in_features=hidden_dim, out_features=2*out_dim)
        self.lin2 = nn.Linear(in_features=2*out_dim, out_features=out_dim)

    def forward(self, data, walks, deg):
        x_0 = self.net(data, walks, deg)
        x_1 = self.lin1(x_0)
        x_1 = F.relu(x_1)
        x_2 = self.lin2(x_1)
        return F.softmax(x_2, dim=-1)


class DBPGCN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True, **kwargs):
        super(DBPGCN, self).__init__(**kwargs)

        self.dropout = dropout

        self.net = DBPnet(num_nodes=num_nodes, in_dim=in_dim, out_dim=hidden_dim, num_walks=num_walks,
                          walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                          transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)

        self.conv1 = GCNConv(in_channels=hidden_dim, out_channels=2 * out_dim, add_self_loops=True)
        self.conv2 = GCNConv(in_channels=2 * out_dim, out_channels=out_dim, add_self_loops=True)

    def forward(self, data, walks, deg):
        x_0 = self.net(data, walks, deg)
        x_1 = self.conv1(x_0, edge_index)
        x_1 = F.relu(x_1)
        x_2 = self.conv2(x_1, edge_index)
        return F.softmax(x_2, dim=-1)


class DBPGAT(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True, **kwargs):
        super(DBPGAT, self).__init__(**kwargs)

        self.dropout = dropout

        self.net = DBPnet(num_nodes=num_nodes, in_dim=in_dim, out_dim=hidden_dim, num_walks=num_walks,
                          walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                          transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)
        self.conv1 = GATConv(in_channels=hidden_dim, out_channels=2 * out_dim, heads=8)
        self.conv2 = GATConv(in_channels=8 * 2 * out_dim, out_channels=out_dim, heads=1)

    def forward(self, data, walks, deg):
        x_0 = self.net(data, walks, deg)
        x_1 = self.conv1(x_0, edge_index)
        x_1 = F.relu(x_1)
        x_2 = self.conv2(x_1, edge_index)
        return F.softmax(x_2, dim=-1)


class DBPSAGE(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True, **kwargs):
        super(DBPSAGE, self).__init__(**kwargs)

        self.dropout = dropout

        self.net = DBPnet(num_nodes=num_nodes, in_dim=in_dim, out_dim=hidden_dim, num_walks=num_walks,
                          walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                          transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)

        self.conv1 = SAGEConv(in_channels=hidden_dim, out_channels=2 * out_dim, normalize=True)
        self.conv2 = SAGEConv(in_channels=2 * out_dim, out_channels=out_dim, normalize=True)

    def forward(self, data, walks, deg):
        x_0 = self.net(data, walks, deg)
        x_1 = self.conv1(x_0, edge_index)
        x_1 = F.relu(x_1)
        x_2 = self.conv2(x_1, edge_index)
        return F.softmax(x_2, dim=-1)


class DBPGIN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim,
                 num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads,
                 dropout, aggr, use_se=True, **kwargs):
        super(DBPGIN, self).__init__(**kwargs)

        self.dropout = dropout

        self.net = DBPnet(num_nodes=num_nodes, in_dim=in_dim, out_dim=hidden_dim, num_walks=num_walks,
                          walks_length=walks_length, walks_heads=walks_heads, num_layers=num_layers,
                          transformer_heads=transformer_heads, dropout=dropout, aggr=aggr, use_se=use_se)
        self.mlp = nn.Sequential(nn.Linear(num_layers*hidden_dim, 2 * out_dim),
                                 nn.ReLU(True),
                                 nn.Linear(2 * out_dim, out_dim))

        self.gin = GINConv(self.mlp, train_eps=True)

    def forward(self, data, walks, deg):
        x_0 = self.net(data, walks, deg)
        x_1 = self.conv1(x_0, edge_index)
        x_out = self.gin(x_1, edge_index)
        return F.softmax(x_out, dim=-1)
