import os
import json

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim

from app.config import TRAINING_DEFAULTS as training_defaults
from app.config import DATA_DEFAULTS as data_defaults
from app.utils.model_registry import get_model_registry
from app.utils.losses import get_criterion

def train_model(dataset, training_config, data_config, verbose=True):

    ## - Override Default Configurations - ##
    training_config = {**training_defaults, **training_config}
    data_config = {**data_defaults, **data_config}

    ## - Load Data - ##
    train_loader, val_loader = load_data(dataset, training_config)

    ## - Get Model Class for Training - ##
    ModelClass = get_model_registry().get(training_config.get('model_name'))

    ## - Build Model - ##
    model = ModelClass(
        task=training_config.get('task'),
        model_name=training_config.get('model_name'),
        
        in_channels=len(dataset[0].x[0]),
        hidden_channels=training_config.get('hidden_channels'),
        out_channels=get_out_channels(training_config.get('task')),

        dropout=training_config.get('dropout'),
        num_layers=training_config.get('num_layers'),
        pooling_method=training_config.get('pooling_method'),

        ## - GNN Specific - ##
        edge_dim=dataset[0].edge_attr.size()[1],
        num_node_types=dataset.num_node_types,
        num_edge_types=dataset.num_edge_types,
        edge_type_emb_dim=training_config.get('edge_type_emb_dim'),
        edge_attr_emb_dim=training_config.get('edge_attr_emb_dim')
    )

    ## - Run Training - ##
    model, results = run_training(model, train_loader, val_loader, training_config)

    ## - Save Model and Configurations - ##
    save_config(training_config, dataset)

    return model, results

def run_training(model, train_loader, val_loader, training_config, verbose=True):
    epochs = training_config.get('epochs')
    opt = optim.Adam(model.parameters(), lr=training_config.get('learning_rate'))
    criterion = get_criterion(training_config.get('task'))

    # Configure model checkpointing
    checkpointing = ModelCheckpoint(
        "checkpoints/"+training_config.get("title"),  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=False,  # Never save the most recent checkpoint, even if it's not the best
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=training_config.get("early_stopping_patience"), mode="min")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        min_epochs=50, # MIN number of epochs to train for
        max_epochs=epochs, # MAX number of epochs to train for
        callbacks=[checkpointing, early_stop], # Use the configured checkpoint callback
    )

    trainer.fit(model, train_loader, val_loader)

    results = trainer.test(dataloaders=val_loader, ckpt_path='best')

    return model, results

def load_data(dataset, training_config):
    data_size = len(dataset)
    train_test_split = training_config.get('train_test_split')

    train_loader = DataLoader(dataset[:int(data_size * train_test_split)],
                        batch_size=training_config.get('batch_size'),
                        shuffle=True)

    if training_config.get('task') == 'ranking':
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        test_loader = DataLoader(dataset[int(data_size * train_test_split):], batch_size=len(dataset[int(data_size * train_test_split):]), shuffle=True)

    return train_loader, test_loader

def get_out_channels(task):
    if task == 'binary_classification':
        return 1
    elif task == 'regression':
        return 1
    elif task == 'ranking':
        return 1

def save_model_from_checkpoint(training_config):
    model = torch.load(f'model_checkpoints/{training_config.get("title")}_checkpoint.pt')
    os.rmdir('model_checkpoints')
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs(training_config.get("title"))
    torch.save(model.state_dict(), f'trained_models/f{training_config.get("title")}/{training_config.get("title")}_Model.pt')
    return model

def save_config(training_config, data_config):
    os.makedirs('checkpoints/'+training_config.get("title"), exist_ok=True)
    with open(f'checkpoints/{training_config.get("title")}/{training_config.get("title")}_training_config.json', 'w') as f:
        json.dump(training_config, f)
    with open(f'checkpoints/{training_config.get("title")}/{training_config.get("title")}_data_config.json', 'w') as f:
        json.dump(data_config, f)
