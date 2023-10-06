import torch
import torch.nn as nn
from torch.nn import LayerNorm

from HPGT.utils.module import se_coder, GraphTransformerLayer
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot


class GraphTransformer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim, num_walks, walks_length, walks_heads,
                 num_layers, transformer_heads, dropout, aggr, use_se=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_walks = num_walks
        self.walk_length = walks_length
        self.walks_heads = walks_heads
        self.num_layers = num_layers
        self.transformer_heads = transformer_heads
        self.dropout = dropout
        self.use_se = use_se

        if self.use_se:
            self.structure_encoder = se_coder(in_dim=in_dim, num_walks=num_walks, walk_length=walks_length,
                                              heads=walks_heads, aggr=aggr)
        self.layer_norm = nn.LayerNorm(out_dim)

        if use_se:
            self.lin = nn.Linear(in_features=2 * in_dim, out_features=out_dim)
        else:
            self.lin = nn.Linear(in_features=in_dim, out_features=out_dim)

        self.transformer_blocks = nn.Sequential()
        for i in range(num_layers):  # 添加transformer-encoder块 一共layers个
            self.transformer_blocks.add_module('block' + str(i + 1),
                                               GraphTransformerLayer(in_dim=out_dim,
                                                                     out_dim=out_dim,
                                                                     heads=transformer_heads, dropout=dropout))
        # self.lin_cat = nn.Linear(in_features=(num_layers + 1) * out_dim, out_features=out_dim)
        self.reset_parameter()

    def forward(self, x, walks, deg):
        if self.use_se:
            se = self.structure_encoder(walks)
            x_cat = torch.cat((se, x), dim=-1)
            x_cat = F.dropout(x_cat, p=self.dropout, training=self.training)
            input_x = self.lin(x_cat)
        else:
            input_x = self.lin(x)
        output_x = None
        # output_list = []
        # output_list.append(input_x)

        for i, blk in enumerate(self.transformer_blocks):
            output_x = blk(input_x, deg)
            # output_list.append(output_x)
            input_x = output_x

        # concat_layer_output = torch.cat(output_list, dim=-1)
        # concat_layer_output = F.relu(concat_layer_output)
        # out = self.lin_cat(concat_layer_output)
        out = F.dropout(output_x, p=self.dropout, training=self.training)
        return out

    def reset_parameter(self):
        self.lin.reset_parameters()
