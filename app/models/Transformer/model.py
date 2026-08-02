from lightning import pytorch as pl
import torch
from torch import nn
from torch_geometric.data import Data
import math

from app.models.ModelMixins.BaseModel import PytorchBaseModel
from app.models.ModelMixins.foundation_model import fm_sequence_to_embedding
from app.models.ModelMixins.steps_mixin import SharedStepsMixin
from app.models.ModelMixins.data_module import collate_token_batch
from app.utils.losses import get_criterion

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer ensures:
        # - moves with .to(device)
        # - saved in state_dict
        # - not a trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]
    

class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2, attn = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # store for hooks / interpretability
        self.attn_weights = attn

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerBackbone(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, vocab_size, dropout, out_dim, features_dim=0, pad_idx=0):
        super().__init__()

        # TODO: Minimal model_dim should be 4
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = TransformerEncoderLayerWithWeights(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

         # Regressor head is now part of the backbone
        reg_in_dim = model_dim + features_dim
        self.regressor = nn.Sequential(
            nn.LayerNorm(reg_in_dim),
            nn.Linear(reg_in_dim, reg_in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reg_in_dim // 2, reg_in_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(reg_in_dim // 4, out_dim)
        )

    def forward(self, token_ids, features=None, attention_mask=None):
        token_ids = token_ids.long()

        # Hard fail before embedding to avoid opaque CUDA device-side assert
        tmin = int(token_ids.min().item())
        tmax = int(token_ids.max().item())
        if tmin < 0 or tmax >= self.vocab_size:
            raise ValueError(
                f"token_ids out of range for embedding: min={tmin}, max={tmax}, "
                f"allowed=[0, {self.vocab_size - 1}], vocab_size={self.vocab_size}"
            )
        
        x = self.embedding(token_ids)
        cls = self.cls_token.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoder(x)

        key_padding_mask = None
        if attention_mask is not None:
            cls_valid = torch.ones(
                (attention_mask.size(0), 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attn_with_cls = torch.cat([cls_valid, attention_mask], dim=1)  # [B, S+1]
            key_padding_mask = ~attn_with_cls.bool()  # True = pad positions

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = x[:, 0]  # CLS token output
        
        if features is not None:
            x = torch.cat([x, features], dim=1)
        return self.regressor(x)


class TransformerLightning(SharedStepsMixin, pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.task = self.hparams.task

        self.backbone = TransformerBackbone(
            model_dim=self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            num_layers=self.hparams.num_layers,
            vocab_size=self.hparams.vocab_size,
            dropout=self.hparams.dropout,
            out_dim=self.hparams.out_dim,
            features_dim=self.hparams.features_dim,
            pad_idx=getattr(self.hparams, "pad_idx", 0)
        )

        self.loss_fn, self.loss_name = get_criterion(self.hparams.task)
        self.setup_metrics()

    def forward(self, batch):
        return self.backbone(batch.token_ids, getattr(batch, "features", None), getattr(batch, "attention_mask", None))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

class TransformerModel(PytorchBaseModel):
    def __init__(self, config, task="regression", experiment_name="transformer", run_name="run"):
        super().__init__(config, task, experiment_name, run_name)
        self.model_type = "transformer"
        self.model = TransformerLightning(**config.__dict__)
        self.backbone = self.model.backbone

    def _predict_helper(self, batch):
        return self.backbone(
            batch.token_ids, 
            getattr(batch, "features", None), 
            getattr(batch, "attention_mask", None)
        )

    def _convert_json_to_pyg(self, sample: dict) -> Data:
        """
        Convert a single Knox JSON sample into a PyTorch Geometric Data object.
        """

        # -------------------------
        # Required: x (sequence tokens)
        # -------------------------
        token_ids = torch.tensor(sample.token_ids, dtype=torch.long)

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

    def _get_collate_fn(self):
        return collate_token_batch
