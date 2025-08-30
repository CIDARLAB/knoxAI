import torch.nn as nn
from torch_geometric.nn import GraphConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class GraphConvBase(nn.Module):
    def __init__(self, **kwargs):
        super(GraphConvBase, self).__init__()

        in_channels = kwargs.get('in_channels')
        hidden_channels = kwargs.get('hidden_channels')
        out_channels = kwargs.get('out_channels')
        self.dropout = kwargs.get('dropout')
        num_layers = kwargs.get('num_layers')
        pooling_method = kwargs.get('pooling_method')

        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}.get(pooling_method)

        # message-passing
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_ch, hidden_channels))

        # post-message-passing
        self.post_mp = nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels), nn.Dropout(self.dropout),
        nn.Linear(hidden_channels, out_channels))

    def forward(self, batch):
        x = batch.x

        # Message passing through all layers
        for conv in self.convs:
            x = conv(x, batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pooling(x, batch.batch)
        x = self.post_mp(x)
        return x
    