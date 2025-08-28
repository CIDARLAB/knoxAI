import torch.nn as nn
from torch_geometric.nn import NNConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class NNConvBase(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout):
        super(NNConvBase, self).__init__()
        
        # message-passing
        edge_mlp1 = nn.Linear(edge_dim, in_channels * hidden_channels)
        self.conv1 = NNConv(in_channels, hidden_channels, nn=edge_mlp1)
        
        edge_mlp2 = nn.Linear(edge_dim, hidden_channels * hidden_channels)
        self.conv2 = NNConv(hidden_channels, hidden_channels, nn=edge_mlp2)
        self.conv3 = NNConv(hidden_channels, hidden_channels, nn=edge_mlp2)
        
        # post-message-passing
        self.post_mp = nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels), nn.Dropout(dropout),
        nn.Linear(hidden_channels, out_channels))

        self.dropout=dropout

        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}

    def forward(self, batch, pooling_method="mean"):
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # First Conv Layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Conv Layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third Conv Layer
        x = self.conv3(x, edge_index, edge_attr)
        emb = x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling[pooling_method](x, batch)
        x = self.post_mp(x)
        return emb, x 
    