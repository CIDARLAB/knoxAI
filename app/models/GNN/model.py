from lightning import pytorch as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphConv, NNConv, GATv2Conv, TransformerConv
from torch_geometric.data import Data

from app.models.ModelMixins.BaseModel import PytorchBaseModel
from app.models.ModelMixins.foundation_model import fm_sequence_to_embedding
from app.models.ModelMixins.steps_mixin import SharedStepsMixin
from app.utils.losses import get_criterion

AGGR_MAP = {
    "mean": pyg_nn.aggr.MeanAggregation,
    "min": pyg_nn.aggr.MinAggregation,
    "max": pyg_nn.aggr.MaxAggregation,
    "sum": pyg_nn.aggr.SumAggregation,
}

class GNN(nn.Module):
    def __init__(
            self,
            hidden_dim : int,
            out_dim: int,
            node_features_dim: int,
            edge_dim: int,
            features_dim: int,
            vocab_size: int,
            embedding_dim: int,
            graph_conv: str,
            aggr: str,
            num_layers: int,
            dropout: float,
        ):
        super().__init__()
        self.dropout = dropout
        self.graph_conv = graph_conv
        in_dim = node_features_dim + embedding_dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # message-passing
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_dim if i == 0 else hidden_dim
            self.convs.append(self.make_graph_conv(in_dim, hidden_dim, edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Aggregation
        self.aggr = AGGR_MAP[aggr]()

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim + features_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, out_dim)
        )

    def forward(
        self, 
        edge_index : Tensor, 
        node_features: Tensor | None = None, 
        node_labels: Tensor | None = None, 
        edge_attr: Tensor | None = None, 
        features: Tensor | None = None, 
        batch: Tensor | None = None
        ):

        # Initial node features
        if node_features is None and node_labels is not None:
            x = self.embedding(node_labels.squeeze(-1) if node_labels.dim() > 1 else node_labels)
        
        elif node_labels is not None:
            label_emb = self.embedding(node_labels.squeeze(-1) if node_labels.dim() > 1 else node_labels)
            x = torch.cat([node_features, label_emb], dim=-1)

        else:
            x = node_features

        # Message passing through all layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = self.use_graph_conv(conv, x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global aggregation
        x = self.aggr(x, batch)
        
        # Concatenate global features
        x = torch.cat([x, features], dim=1) if features is not None else x
        
        # Post-message-passing MLP
        return self.post_mp(x)

    def make_graph_conv(self, in_dim : int, hidden_dim: int, edge_dim: int | None = None):
        
        if self.graph_conv.lower() == "graph":
            return GraphConv(in_dim, hidden_dim)

        elif self.graph_conv.lower() == 'nn':
            edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.ReLU(),
                nn.Linear(64, in_dim * hidden_dim)
            )
            return NNConv(in_dim, hidden_dim, nn=edge_mlp)

        elif self.graph_conv.lower() == 'gat':
            return GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim)

        elif self.graph_conv.lower() == 'tconv':
            return TransformerConv(in_dim, hidden_dim, edge_dim=edge_dim)

    def use_graph_conv(self, conv, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        
        if self.graph_conv.lower() == "graph":
            return conv(x, edge_index)

        elif self.graph_conv.lower() == 'nn':
            return conv(x, edge_index, edge_attr)

        elif self.graph_conv.lower() == 'gat':
            return conv(x, edge_index, edge_attr)

        elif self.graph_conv.lower() == 'tconv':
            return conv(x, edge_index, edge_attr)


class GNNLightning(SharedStepsMixin, pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.task = self.hparams.task

        self.backbone = GNN(
            self.hparams.hidden_dim,
            self.hparams.out_dim,
            self.hparams.node_features_dim,
            self.hparams.edge_dim,
            self.hparams.features_dim,
            self.hparams.vocab_size,
            self.hparams.embedding_dim,
            self.hparams.graph_conv,
            self.hparams.aggr,
            self.hparams.num_layers,
            self.hparams.dropout,
        )

        self.loss_fn, self.loss_name = get_criterion(self.hparams.task)
        self.setup_metrics()

    def forward(self, batch):
        return self.backbone(
            batch.edge_index,
            getattr(batch, 'node_features', None),
            getattr(batch, 'node_labels', None),
            getattr(batch, 'edge_attr', None),
            getattr(batch, 'features', None),
            getattr(batch, 'batch', None),
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

class GNNModel(PytorchBaseModel):
    def __init__(self, config, task="regression", experiment_name="gnn", run_name="run"):
        super().__init__(config, task, experiment_name, run_name)
        self.model_type = "gnn"
        self.model = GNNLightning(**config.__dict__)
        self.backbone = self.model.backbone
        self.data_point_printed = False

    def _predict_helper(self, batch):
        return self.backbone(
            batch.edge_index,
            batch.node_features if hasattr(batch, 'node_features') else None,
            batch.node_labels if hasattr(batch, 'node_labels') else None,
            batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            batch.features if hasattr(batch, 'features') else None,
            batch.batch if hasattr(batch, 'batch') else None,
        )

    def _convert_json_to_pyg(self, sample: dict) -> Data:
        """
        Convert a single Knox JSON sample into a PyTorch Geometric Data object.
        """

        # -------------------------
        # Required: edge_index
        # -------------------------
        edge_index = torch.tensor(sample.edge_index, dtype=torch.long)

        # -------------------------
        # Optional: node_features
        # -------------------------
        node_features = torch.tensor(sample.node_features, dtype=torch.float) if sample.node_features is not None else None

        # -------------------------
        # Optional: edge_attr (edge features)
        # -------------------------
        edge_attr = torch.tensor(sample.edge_attr, dtype=torch.float) if sample.edge_attr is not None else None

        # -------------------------
        # Optional: node_label (categorical node labels)
        # -------------------------
        node_labels = torch.tensor(sample.node_labels, dtype=torch.long) if sample.node_labels is not None else None

        # -------------------------
        # Optional: foundation model sequence embeddings (node-level)
        # -------------------------
        node_seq_embeddings = torch.tensor([fm_sequence_to_embedding(seq) for seq in sample.node_sequence], dtype=torch.float) if sample.node_sequence is not None else None

        # -------------------------
        # Optional: edge_label (categorical edge labels)
        # -------------------------
        edge_labels = torch.tensor(sample.edge_labels, dtype=torch.long) if sample.edge_labels is not None else None

        # -------------------------
        # Optional: additional design-level features
        # -------------------------
        features = torch.tensor(sample.features, dtype=torch.float) if sample.features is not None else None

        # -------------------------
        # Optional: foundation model sequence embeddings (design-level)
        # -------------------------
        design_seq_embeddings = torch.tensor(fm_sequence_to_embedding(sample.sequence), dtype=torch.float) if sample.sequence is not None else None

        # -------------------------
        # Optional: label y
        # -------------------------
        if self.task == "multiclass_classification":
            y = torch.tensor(sample.y, dtype=torch.long) if sample.y is not None else None
        else:
            y = torch.tensor(sample.y, dtype=torch.float) if sample.y is not None else None

        # -------------------------
        # Build PyG Data object
        # -------------------------
        data = Data(
            node_features=node_features,
            y=y,
            features=features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_labels=node_labels,
            edge_labels=edge_labels,
            node_seq_embeddings=node_seq_embeddings,
            design_seq_embeddings=design_seq_embeddings,
        )

        if self.data_point_printed == False:
            print("\n\nConstructed PyG Data object:")
            print(data)
            print("\n")
            self.data_point_printed = True

        return data
    