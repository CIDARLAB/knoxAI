from lightning import pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphConv, NNConv, GATv2Conv, HEATConv, TransformerConv

from app.models.GNN.steps import SharedStepsMixin
from app.utils.losses import get_criterion


class GNN(SharedStepsMixin, pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.task = kwargs.get('task', 'regression')
        self.graph_conv = kwargs.get('model_name')
        self.loss_fn, self.loss_name = get_criterion(self.task)

        self.setup_metrics()
        
        in_channels = kwargs.get('in_channels')
        self.hidden_channels = kwargs.get('hidden_channels')
        out_channels = kwargs.get('out_channels')
        self.dropout = kwargs.get('dropout')
        num_layers = kwargs.get('num_layers')
        pooling_method = kwargs.get('pooling_method', 'mean')
        num_global_features = kwargs.get('num_global_features', 0)

        self.edge_dim = kwargs.get('edge_dim')
        self.num_node_types = kwargs.get('num_node_types', 2)
        self.num_edge_types = kwargs.get('num_edge_types', 3)
        self.edge_type_emb_dim = kwargs.get('edge_type_emb_dim', 4)
        self.edge_attr_emb_dim = kwargs.get('edge_attr_emb_dim', 4)
        
        # message-passing
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else self.hidden_channels
            self.convs.append(self.make_graph_conv(in_ch))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(self.hidden_channels + num_global_features, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels // 2, out_channels)
        )
        
        self.pooling = {"mean": pyg_nn.global_mean_pool,
                        "add" : pyg_nn.global_add_pool,
                        "max" : pyg_nn.global_max_pool}.get(pooling_method)

    def forward(self, batch):
        x = batch.x

        # Message passing through all layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = self.use_graph_conv(conv, x, batch)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pooling(x, batch.batch)
        
        # Concatenate global features
        if hasattr(batch, 'global_features'):
            x = torch.cat([x, batch.global_features], dim=1)
            
        x = self.post_mp(x)
        return x 
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def make_graph_conv(self, in_channels):
        if self.graph_conv.lower() == "heat":
            return HEATConv(in_channels, self.hidden_channels, self.num_node_types, self.num_edge_types, self.edge_type_emb_dim, self.edge_dim, self.edge_attr_emb_dim)
        
        elif self.graph_conv.lower() == "graph":
            return GraphConv(in_channels, self.hidden_channels)

        elif self.graph_conv.lower() == 'nn':
            edge_mlp = nn.Linear(self.edge_dim, in_channels * self.hidden_channels)
            return NNConv(in_channels, self.hidden_channels, nn=edge_mlp)

        elif self.graph_conv.lower() == 'gat':
            return GATv2Conv(in_channels, self.hidden_channels, edge_dim=self.edge_dim)

        elif self.graph_conv.lower() == 'tconv':
            return TransformerConv(in_channels, self.hidden_channels, edge_dim=self.edge_dim)

    def use_graph_conv(self, conv, x, batch):
        if self.graph_conv.lower() == "heat":
            return conv(x, batch.edge_index, batch.node_type, batch.edge_type, batch.edge_attr)
        
        elif self.graph_conv.lower() == "graph":
            return conv(x, batch.edge_index)

        elif self.graph_conv.lower() == 'nn':
            return conv(x, batch.edge_index, batch.edge_attr)

        elif self.graph_conv.lower() == 'gat':
            return conv(x, batch.edge_index, batch.edge_attr)

        elif self.graph_conv.lower() == 'tconv':
            return conv(x, batch.edge_index, batch.edge_attr)
        