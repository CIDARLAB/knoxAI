from lightning import pytorch as pl
import torch
from torch import nn
from torch_geometric.data import Data

from app.models.ModelMixins.BaseModel import PytorchBaseModel
from app.models.ModelMixins.steps_mixin import SharedStepsMixin
from app.utils.losses import get_criterion
from app.models.ModelMixins.foundation_model import fm_sequence_to_embedding

class MLPBackbone(nn.Module):
    def __init__(
            self,
            sequence_length,
            features_dim,
            embedding_dim, 
            hidden_dim, 
            out_dim, 
            num_layers, 
            dropout, 
            vocab_size, 
            pad_idx=0
        ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        layers = []
        in_dim = features_dim + embedding_dim * sequence_length
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, token_ids=None, features=None):
        if token_ids is not None:
            embedded_tokens = [
                self.embedding(token_ids[:, i])
                for i in range(token_ids.size(1))
            ]
            embedded_tokens = torch.cat(embedded_tokens, dim=-1)

            if features is not None:
                x = torch.cat([features, embedded_tokens], dim=-1)
            else:
                x = embedded_tokens

        elif features is not None:
            x = features

        else:
            raise ValueError("Both token_ids and features are None in MLPBackbone forward pass.")
        
        return self.mlp(x)
    
class MLPLightning(SharedStepsMixin, pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.task = self.hparams.task

        self.backbone = MLPBackbone(
            sequence_length=self.hparams.sequence_length,
            features_dim=self.hparams.features_dim,
            embedding_dim=self.hparams.embedding_dim,
            hidden_dim=self.hparams.hidden_dim,
            out_dim=self.hparams.out_dim,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
            vocab_size=self.hparams.vocab_size
        )

        self.loss_fn, self.loss_name = get_criterion(self.hparams.task)
        self.setup_metrics()

    def forward(self, batch):
        return self.backbone(getattr(batch, "token_ids", None), getattr(batch, "features", None))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
class MLPModel(PytorchBaseModel):
    def __init__(self, config, task="regression", experiment_name="transformer", run_name="run"):
        super().__init__(config, task, experiment_name, run_name)
        self.model_type = "mlp"
        self.model = MLPLightning(**config.__dict__)
        self.backbone = self.model.backbone

    def _predict_helper(self, batch):
        return self.backbone(getattr(batch, "token_ids", None), getattr(batch, "features", None))
    
    def _convert_json_to_pyg(self, sample: dict) -> Data:
        """
        Convert a single Knox JSON sample into a PyTorch Geometric Data object.
        """

        # -------------------------
        # Optional: x (sequence tokens)
        # -------------------------
        token_ids = torch.tensor(sample.token_ids, dtype=torch.long) if sample.token_ids is not None else None

        # -------------------------
        # Optional: additional design-level features
        # -------------------------
        features = torch.tensor(sample.features, dtype=torch.float) if sample.features is not None else None

        # -------------------------
        # Optional: foundation model sequence embeddings (design-level)
        # -------------------------
        seq_embeddings = torch.tensor(fm_sequence_to_embedding(sample.sequence), dtype=torch.float) if sample.sequence is not None else None

        # -------------------------
        # Optional: label y
        # -------------------------
        if self.task in ["classification", "multiclass_classification"]:
            y = torch.tensor(sample.y, dtype=torch.long) if sample.y is not None else None
        else:
            y = torch.tensor(sample.y, dtype=torch.float) if sample.y is not None else None

        # -------------------------
        # Build PyG Data object
        # -------------------------
        data = Data(
            token_ids=token_ids,
            y=y,
            features=features,
            seq_embeddings=seq_embeddings
        )

        return data