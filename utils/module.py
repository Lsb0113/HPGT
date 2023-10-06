import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
import math


class se_coder(nn.Module):
    def __init__(self, in_dim, num_walks, walk_length, heads=1, aggr='mean'):
        super().__init__()
        self.x_dim = in_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.heads = heads
        self.aggr = aggr
        self.self_attn = nn.Parameter(torch.Tensor(self.x_dim, heads))

        self.lin = nn.Linear(in_features=in_dim * heads, out_features=in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.self_attn)

    def forward(self, x_walks):
        x_attn = x_walks @ self.self_attn
        ep = F.leaky_relu(x_attn)
        alpha_p = F.softmax(ep, dim=-2)
        alpha_p_adjust = torch.transpose(input=alpha_p, dim0=0, dim1=-1).reshape(self.heads,
                                                                                 x_walks.shape[0],
                                                                                 self.num_walks, 1,
                                                                                 -1)
        x_attn_fused = F.relu(alpha_p_adjust @ x_walks).reshape(self.heads, x_walks.shape[0],
                                                                self.num_walks,
                                                                -1)

        if self.aggr == 'mean':
            x_attn_aggr = x_attn_fused.mean(-2)
        elif self.aggr == 'sum':
            x_attn_aggr = x_attn_fused.sum(-2)
            x_attn_aggr = x_attn_aggr / (x_attn_aggr.sum(-1).unsqueeze(-1) + 1e-30)
        else:
            x_attn_aggr = x_attn_fused.max(-2).values
        list_x_attn_aggr = [x_attn_aggr[i] for i in range(x_attn_aggr.shape[0])]
        x_attn_cat = torch.cat(list_x_attn_aggr, dim=-1)
        return self.lin(x_attn_cat)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.WQ = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.WK = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.WV = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.lin = nn.Linear(in_features=heads * out_dim, out_features=out_dim)
        self.reset_parameters()

    def forward(self, input_x):
        Q = input_x @ self.WQ
        K = input_x @ self.WK
        V = input_x @ self.WV
        KT = torch.transpose(K, 2, 1)
        QKT = Q @ KT
        for i in range(self.heads):
            diag = torch.diag(QKT[i]) + (-1e20)
            diag_mtx = torch.diag_embed(diag)
            QKT[i] = QKT[i] + diag_mtx
        attn = F.softmax((QKT / math.sqrt(self.out_dim)), dim=-1)
        attn_x = attn @ V
        list_attn_x = [x for x in attn_x]
        attn_x_cat = torch.cat(list_attn_x, dim=1)
        attn_x_lin = self.lin(attn_x_cat)
        return attn_x_lin

    def reset_parameters(self):
        glorot(self.WK)
        glorot(self.WQ)
        glorot(self.WV)
        self.lin.reset_parameters()


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.attn = MultiHeadAttentionLayer(in_dim=in_dim, out_dim=out_dim, heads=heads)
        self.lin = nn.Linear(in_features=2 * out_dim, out_features=out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.fnn_1 = nn.Linear(in_features=out_dim, out_features=out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)
        self.fnn_2 = nn.Linear(in_features=out_dim, out_features=out_dim)

    def forward(self, input_x, deg):
        attn_x = self.attn(input_x)
        x_1 = torch.cat((input_x, attn_x / torch.sqrt(deg).reshape(-1, 1)), dim=-1)
        x_1 = self.lin(x_1)

        x_1 = input_x + x_1
        x_1 = self.layer_norm1(x_1)
        x_1_save = x_1
        # FNN
        x_2 = self.fnn_1(x_1)
        x_2 = F.relu(x_2)
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_2 = self.fnn_2(x_2)
        x_2 = x_1_save + x_2
        out = self.layer_norm2(x_2)

        return out
