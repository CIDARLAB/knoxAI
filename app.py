from flask import Flask, jsonify, request

import torch
from torch_geometric.loader import DataLoader

from utils.graph_dataset import GraphDataset
from training.model_training import train_model
from model_registry import get_model_registry
from utils.metrics import get_available_metrics
from utils.load_config import load_config
from evaluate.evaluate_model import test_model
from predict.model_prediction import predict

app = Flask(__name__)

@app.route('/train-model', methods=["POST"])
def train_model():
    ## - Get Configurations - ##
    config = request.get_json()
    data = config.get('data')
    data_config = config.get('data_config')
    training_config = config.get('training_config')

    ## - Load Dataset - ##
    dataset = GraphDataset(data, training_config.get('task'), data_config, data.get('part_analytics')).get_batch()

    ## - Train Model - ##
    model, best_test_loss, test_metric = train_model(dataset, training_config, verbose=True)

    return jsonify({"best_test_loss": best_test_loss, "test_metric": test_metric})

@app.route('/evaluate-model', methods=["POST"])
def evaluate_model():
    config = request.get_json()
    data = config.get('data')
    training_config, data_config = load_config(config.get('title'))
    model = torch.load(f'training/trained_models/f{training_config.get("title")}/{training_config.get("title")}_Model.pt')

    dataset = GraphDataset(data, training_config.get('task'), data_config, data.get('part_analytics')).get_batch()

    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    loss, metrics = test_model(model, test_loader, training_config, all_metrics=True)

    return jsonify({"loss": loss, "metrics": metrics})

@app.route('/predict-model', methods=["POST"])
def predict_model():
    config = request.get_json()
    data = config.get('data')
    training_config, data_config = load_config(config.get('title'))
    model = torch.load(f'training/trained_models/f{training_config.get("title")}/{training_config.get("title")}_Model.pt')

    dataset = GraphDataset(data, training_config.get('task'), data_config, data.get('part_analytics')).get_batch()

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    prediction = predict(model, loader, training_config)

    return jsonify({"prediction": prediction})

@app.route('/available-metrics', methods=["GET"])
def available_metrics():
    return jsonify(get_available_metrics().keys())

@app.route('/model-registry', methods=["GET"])
def model_registry():
    return jsonify(get_model_registry())

if __name__ == '__main__':
    app.run(debug=True)
