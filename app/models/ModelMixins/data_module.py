import mlflow
from mlflow.data import from_pandas
import pandas as pd
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0

def log_dataset(name: str, json_list: list[dict]):
    print(f"{name} dataset logged with {len(json_list)} samples.")
    print(json_list[0])
    print()
    json_list = convert_values_to_str(json_list)
    df = pd.DataFrame(json_list)
    dataset = from_pandas(df, name=name)
    mlflow.log_input(dataset, context=name)

def convert_values_to_str(data: list[dict]) -> list[dict]:
    converted_data = []
    for sample in data:
        converted_sample = {}
        for key, value in sample.__dict__.items():
            converted_sample[key] = str(value)
        converted_data.append(converted_sample)
    return converted_data

def _normalize_token_ids(x):
    # Unwrap singleton nesting like [[...]] or [[[...]]]
    while isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]

    t = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)
    t = t.to(dtype=torch.long)

    # Unwrap tensor singleton dims: [1, L] or [L, 1] -> [L]
    while t.ndim > 1 and 1 in t.shape:
        t = t.squeeze()

    if t.ndim != 1:
        raise ValueError(
            f"token_ids must resolve to 1D per sample, got shape={tuple(t.shape)}"
        )

    return t.contiguous()

def collate_token_batch(batch):
    token_ids = [_normalize_token_ids(b.token_ids) for b in batch]
    token_ids = pad_sequence(token_ids, batch_first=True, padding_value=PAD_IDX).long()

    if torch.any(token_ids < 0):
        bad = token_ids[token_ids < 0][:10].tolist()
        raise ValueError(f"Negative token_ids found (first few): {bad}")

    attention_mask = (token_ids != PAD_IDX)

    features = None
    if getattr(batch[0], "features", None) is not None:
        features = torch.stack([b.features for b in batch], dim=0)

    y = None
    if getattr(batch[0], "y", None) is not None:
        y = torch.stack([b.y for b in batch], dim=0)
        if y.ndim == 2 and y.size(-1) == 1:   # [B,1] -> [B]
            y = y.squeeze(-1)

    return Data(
        token_ids=token_ids,
        attention_mask=attention_mask,
        features=features,
        y=y,
    )

class PytorchDataModule(LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        test_data,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        collate_fn=None
    ):
        super().__init__()

        # These are lists of torch_geometric.data.Data objects
        self.train_data = train_data
        self.val_data   = val_data
        self.test_data  = test_data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        # Optional sanity check
        if len(train_data) > 0:
            assert isinstance(train_data[0], Data), \
                "train_data must be a list of torch_geometric.data.Data objects"

    # Lightning calls these automatically
    def train_dataloader(self):
        return self._make_loader(self.train_data, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._make_loader(self.val_data, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_data, shuffle=False)

    def _make_loader(self, data, shuffle=False):
        if self.collate_fn is None:
            return PyGDataLoader(
                data,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        return TorchDataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )