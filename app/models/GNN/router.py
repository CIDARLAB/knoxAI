from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

import torch
from torch_geometric.loader import DataLoader

from app.utils.graph_dataset import GraphDataset
from app.models.GNN.train import train_model
from app.models.GNN.evaluate import test_model
from app.models.GNN.predict import predict
from app.utils.load_config import load_config
from app.config import DATA_DEFAULTS as data_DEFAULTS
from app.config import TRAINING_DEFAULTS as training_DEFAULTS

router = APIRouter(prefix="/gnn", tags=["GNN"])

@router.post('/train-model')
def train_model():
    try:
        ## - Get Configurations - ##
        config = router.current_request.json_body
        data = config.get('data')
        data_config = config.get('data_config')
        training_config = config.get('training_config')

        ## - Load Dataset - ##
        dataset = GraphDataset(data, training_config.get('task'), data_config).get_batch()

        ## - Train Model - ##
        model, val_results = train_model(dataset, training_config, verbose=True)

        return JSONResponse(content=jsonable_encoder({"validation_results": val_results[0]}))
    
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@router.post('/evaluate-model')
def evaluate_model():
    try:
        config = router.current_request.json_body
        data = config.get('data')
        training_config, data_config = load_config(config.get('title'))
        model = torch.load(f'training/trained_models/f{training_config.get("title")}/{training_config.get("title")}_Model.pt')

        dataset = GraphDataset(data, training_config.get('task'), data_config).get_batch()

        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        loss, metrics = test_model(model, test_loader, training_config, all_metrics=True)

        return JSONResponse(content=jsonable_encoder({"loss": loss, "metrics": metrics}))
    
    except FileNotFoundError as e:
        return JSONResponse(content=jsonable_encoder({"error": str(e)}), status_code=404)
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@router.post('/predict-model')
def predict_model():
    try:
        config = router.current_request.json_body
        data = config.get('data')
        training_config, data_config = load_config(config.get('title'))
        model = torch.load(f'training/trained_models/f{training_config.get("title")}/{training_config.get("title")}_Model.pt')

        dataset = GraphDataset(data, training_config.get('task'), data_config).get_batch()

        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        prediction = predict(model, loader, training_config)

        return JSONResponse(content=jsonable_encoder({"prediction": prediction}))

    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@router.get('/config')
def config():
    try:
        ## - Get Default Configurations - ##
        return JSONResponse(content=jsonable_encoder({"data_config": data_DEFAULTS, "training_config": training_DEFAULTS}))
    except Exception as e:
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)
    