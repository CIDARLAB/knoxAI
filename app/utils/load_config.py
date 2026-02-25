import os
import json

def load_config(title):
    training_path = f'config/{title}_training_config.json'
    data_path = f'config/{title}_data_config.json'

    if not os.path.isfile(training_path):
        raise FileNotFoundError(f"Training config file not found: {training_path}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data config file not found: {data_path}")

    with open(training_path, 'r') as f:
        training_config = json.load(f)

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    return training_config, data_config
