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

        # message-passing
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

        # post-message-passing
        self.post_mp = nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels), nn.Dropout(self.dropout),
        nn.Linear(hidden_channels, out_channels))

        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}

    def forward(self, batch, pooling_method="mean"):
        x = batch.x
        edge_index = batch.edge_index
        batch = batch.batch

        # First Conv Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Conv Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third Conv Layer
        x = self.conv3(x, edge_index)
        emb = x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling[pooling_method](x, batch)
        x = self.post_mp(x)
        return emb, x
    