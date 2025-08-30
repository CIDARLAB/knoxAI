import torch.nn as nn
from torch_geometric.nn import HEATConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class HEATConv(nn.Module):
    """
    HEATConv

    A flexible Graph Neural Network (GNN) model using HEATConv layers from PyTorch Geometric.
    This model supports heterogeneous graphs with multiple node and edge types, as well as rich node and edge attributes.
    It is designed for complex graph structures where both node and edge information are important for learning.

    Use this model when:
        - Your graph data contains multiple node types and/or edge types.
        - You want to incorporate node and edge attributes into message passing.
        - You are working with heterogeneous graphs or graphs with complex relational information.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in each layer.
        out_channels (int): Number of output units.
        num_node_types (int): Number of unique node types.
        num_edge_types (int): Number of unique edge types.
        edge_type_emb_dim (int): Embedding dimension for edge types.
        edge_attr_emb_dim (int): Embedding dimension for edge attributes.
        edge_dim (int): Number of edge attribute features.
        num_layers (int): Number of HEATConv layers.
        dropout (float): Dropout probability.
        pooling_method (str): Pooling method to use ('mean', 'add', or 'max').

    """
    
    def __init__(self, **kwargs):
        super(HEATConv, self).__init__()

        in_channels = kwargs.get('in_channels')
        hidden_channels = kwargs.get('hidden_channels')
        out_channels = kwargs.get('out_channels')
        edge_dim = kwargs.get('edge_dim')
        self.dropout = kwargs.get('dropout')
        num_node_types = kwargs.get('num_node_types')
        num_edge_types = kwargs.get('num_edge_types')
        edge_type_emb_dim = kwargs.get('edge_type_emb_dim')
        edge_attr_emb_dim = kwargs.get('edge_attr_emb_dim')
        num_layers = kwargs.get('num_layers')
        pooling_method = kwargs.get('pooling_method')

        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}.get(pooling_method)
        
        # message-passing
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(
                HEATConv(
                    in_ch, hidden_channels, num_node_types, num_edge_types,
                    edge_type_emb_dim, edge_dim, edge_attr_emb_dim
                )
            )
        
        # post-message-passing
        self.post_mp = nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels), nn.Dropout(self.dropout),
        nn.Linear(hidden_channels, out_channels))

    def forward(self, batch):
        x = batch.x
        
        # Message passing through all layers
        for conv in self.convs:
            x = conv(x, batch.edge_index, batch.node_type, batch.edge_type, batch.edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling(x, batch.batch)
        x = self.post_mp(x)
        return x
    