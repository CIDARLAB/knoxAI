import torch.nn as nn
from torch_geometric.nn import NNConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

class NNConvBase(nn.Module):
    """
    NNConvBase

    A Graph Neural Network (GNN) model using NNConv layers from PyTorch Geometric.
    This model is designed for graphs where edge attributes play a significant role in message passing.
    NNConv uses neural networks to dynamically generate convolutional filters based on edge features.

    Use this model when:
        - Your graph data includes important edge attributes or features.
        - You want to leverage edge-conditioned message passing for more expressive modeling.
        - You are working with graphs where relationships between nodes are characterized by rich edge information.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units in each layer.
        out_channels (int): Number of output units.
        edge_dim (int): Number of edge attribute features.
        num_layers (int): Number of NNConv layers.
        dropout (float): Dropout probability.
        pooling_method (str): Pooling method to use ('mean', 'add', or 'max').

    """
    
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get('in_channels')
        hidden_channels = kwargs.get('hidden_channels')
        out_channels = kwargs.get('out_channels')
        edge_dim = kwargs.get('edge_dim')
        self.dropout = kwargs.get('dropout')
        num_layers = kwargs.get('num_layers')
        pooling_method = kwargs.get('pooling_method')

        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}.get(pooling_method)

        # message-passing
        edge_mlp1 = nn.Linear(edge_dim, in_channels * hidden_channels)
        edge_mlp2 = nn.Linear(edge_dim, hidden_channels * hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            edge_mlp = edge_mlp1 if i == 0 else edge_mlp2
            self.convs.append(NNConv(in_ch, hidden_channels, nn=edge_mlp))

        # post-message-passing
        self.post_mp = nn.Sequential(
        nn.Linear(hidden_channels, hidden_channels), nn.Dropout(self.dropout),
        nn.Linear(hidden_channels, out_channels))

    def forward(self, batch):
        x = batch.x
        
        # Message passing through all layers
        for conv in self.convs:
            x = conv(x, batch.edge_index, batch.edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pooling(x, batch.batch)
        x = self.post_mp(x)
        return x 
    