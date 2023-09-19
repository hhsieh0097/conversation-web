import torch
import torch.nn as nn
from torch.autograd import Variable

import torch_geometric
from torch_geometric.nn import GCNConv, global_add_pool


class TokenNetwork(nn.Module):
    def __init__(
            self, 
            node_feature_dim, 
            hidden_channel_dim, 
            dropout_rate, 
            device
        ):
        super(TokenNetwork, self).__init__()

        self.gconv_1 = GCNConv(in_channels=node_feature_dim, out_channels=hidden_channel_dim, add_self_loops=False)
        self.gconv_2 = GCNConv(in_channels=hidden_channel_dim, out_channels=hidden_channel_dim, add_self_loops=False)
        self.gconv_3 = GCNConv(in_channels=hidden_channel_dim, out_channels=hidden_channel_dim, add_self_loops=False)

        self.dropout_layer = nn.Dropout(dropout_rate)
        
        self.device = device

    def graphity(self, graphs):
        graphs = [sample for sample in torch_geometric.loader.DataLoader(graphs, batch_size=len(graphs), shuffle=False)][0]
        gx, edge_index, batch, edge_attr = graphs.x, graphs.edge_index, graphs.batch, graphs.edge_attr

        gx = Variable(gx, requires_grad=True).to(self.device)
        edge_index = Variable(edge_index, requires_grad=False).to(self.device)
        batch = Variable(batch, requires_grad=False).to(self.device)
        edge_attr = Variable(edge_attr, requires_grad=True).to(self.device)
        
        return gx, edge_index, batch, edge_attr

    def forward(self, graphs):
        gx, edge_index, batch, edge_attr = self.graphity(graphs)

        gx = self.dropout_layer(torch.relu(self.gconv_1(gx, edge_index, edge_attr)))
        gx = self.dropout_layer(torch.relu(self.gconv_2(gx, edge_index, edge_attr)))
        gx = self.gconv_3(gx, edge_index, edge_attr)

        readout_x = self.dropout_layer(global_add_pool(gx, batch))

        return gx, readout_x