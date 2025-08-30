import torch.nn as nn
from torch_geometric.nn import HEATConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class HEATConvBase(nn.Module):
    def __init__(self, **kwargs):
        super(HEATConvBase, self).__init__()

        in_channels = kwargs.get('in_channels')
        hidden_channels = kwargs.get('hidden_channels')
        out_channels = kwargs.get('out_channels')
        edge_dim = kwargs.get('edge_dim')
        self.dropout = kwargs.get('dropout')
        num_node_types = kwargs.get('num_node_types')
        num_edge_types = kwargs.get('num_edge_types')
        edge_type_emb_dim = kwargs.get('edge_type_emb_dim')
        edge_attr_emb_dim = kwargs.get('edge_attr_emb_dim')
        
        # message-passing
        self.conv1 = HEATConv(in_channels, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim)
        self.conv2 = HEATConv(hidden_channels, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim)
        self.conv3 = HEATConv(hidden_channels, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim)
        
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
        node_type = batch.node_type
        edge_type = batch.edge_type
        edge_attr = batch.edge_attr
        batch = batch.batch
        
        # First Conv Layer
        x = self.conv1(x, edge_index, node_type, edge_type, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second Conv Layer
        x = self.conv2(x, edge_index, node_type, edge_type, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third Conv Layer
        x = self.conv3(x, edge_index, node_type, edge_type, edge_attr)
        emb = x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pooling[pooling_method](x, batch)
        x = self.post_mp(x)
        return emb, x
    