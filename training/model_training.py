import os
import json

from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim

from config.training_config import DEFAULTS as training_defaults
from config.data_config import DEFAULTS as data_defaults
from model_registry import get_model_registry
from evaluate.evaluate_model import test_model

def train_model(dataset, training_config, data_config, verbose=True):

    ## - Override Default Configurations - ##
    training_config = {**training_defaults, **training_config}
    data_config = {**data_defaults, **data_config}

    ## - Load Data - ##
    train_loader, test_loader = load_data(dataset, training_config)

    ## - Get Model Class for Training - ##
    ModelClass = get_model_registry().get(training_config.get('task')).get(training_config.get('model_name'))

    ## - Build Model - ##
    model = ModelClass(in_channels=len(dataset[0].x[0]),
                       hidden_channels=training_config.get('hidden_channels'),
                       out_channels=get_out_channels(training_config.get('task')),
                       edge_dim=dataset[0].edge_attr.size()[1])

    ## - Run Training - ##
    model, best_test_loss, test_metric = run_training(model, train_loader, test_loader, training_config)

    ## - Save Model and Configurations - ##
    save_model_from_checkpoint(training_config)
    save_config(training_config, dataset)

    return model, best_test_loss, test_metric

def run_training(model, train_loader, test_loader, training_config, verbose=True):
    epochs = training_config.get('epochs')
    opt = optim.Adam(model.parameters(), lr=training_config.get('learning_rate'))

    best_test_loss = None
    patience = 0
    for epoch in range(epochs):
        total_loss = 0
        total_graphs = 0

        model.train()
        for batch in train_loader:
            opt.zero_grad()
            embedding, pred = model(batch, training_config.get('pooling_method'))
            loss = model.loss(pred, batch.y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_graphs += batch.num_graphs

        average_loss = total_loss / total_graphs

        test_loss, test_metric = test_model(model, test_loader, training_config)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {average_loss:.4f} - "
                  f"Test Loss: {test_loss:.4f} - "
                  f"{training_config.get('training_metric')}: {test_metric:.4f}")

        if best_test_loss is None or test_loss < best_test_loss:
            os.makedirs('model_checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'model_checkpoints/{training_config.get("title")}_checkpoint.pt')
            best_test_loss = test_loss
            best_metric = test_metric
            patience = 0
        else:
            patience += 1

        if patience >= training_config.get('early_stopping_patience'):
            if verbose:
                print("Early stopping triggered.")
            model = torch.load(f'model_checkpoints/{training_config.get("title")}_checkpoint.pt')
            break

    return model, best_test_loss, test_metric

def load_data(dataset, training_config):
    data_size = len(dataset)
    train_test_split = training_config.get('train_test_split')

    train_loader = DataLoader(dataset[:int(data_size * train_test_split)],
                        batch_size=training_config.get('batch_size'),
                        shuffle=True)

    if training_config.get('task') == 'ranking':
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        test_loader = DataLoader(dataset[int(data_size * train_test_split):], batch_size=1, shuffle=True)

    return train_loader, test_loader

def get_out_channels(task):
    if task == 'binary_classification':
        return 2
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
    os.makedirs('config', exist_ok=True)
    with open(f'config/{training_config.get("title")}_training_config.json', 'w') as f:
        json.dump(training_config, f)
    with open(f'config/{training_config.get("title")}_data_config.json', 'w') as f:
        json.dump(data_config, f)
