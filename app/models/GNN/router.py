from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

import logging

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)

from app.utils.schemas import *
from app.models.GNN.model import GNNModel
from app.models.GNN.config import Config
from app.models.ModelMixins.tune import tune, stop_tuning

from app.models.RandomForest.model import RandomForestModel
from app.models.RandomForest.config import Config as RFConfig

router = APIRouter(prefix="/gnn", tags=["GNN"])

@router.post("/train", response_model=TrainResponse)
def train_endpoint(request: GNNTrainRequest):
    request.config["vocab_size"] = request.vocab_size
    request.config["task"] = request.task

    model = GNNModel(
        config=Config(**request.config), 
        task=request.task, 
        experiment_name=request.experiment_name, 
        run_name=request.run_name
    )

    model.train(
        train_json=request.data.train,
        val_json=request.data.val,
        test_json=request.data.test,
        save_model=True
    )

    model.build_surrogate_model(
        model_class=RandomForestModel,
        model_config=RFConfig(),
        train_rule_matrix=request.rule_matrix.x_train,
        test_rule_matrix=request.rule_matrix.x_test,
        train_json=request.data.train,
        val_json=request.data.val,
        test_json=request.data.test,
        rule_names=request.rule_matrix.feature_names

    ) if request.rule_matrix is not None else None

    return TrainResponse(
        run_id=model.run_id, 
        shap_values=model.shap_values.values.tolist() if model.shap_values is not None else None
    )


@router.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: GNNPredictRequest):
    model = GNNModel(config=Config())

    model.load_inference_model(request.run_id)

    return PredictResponse(predictions=model.predict(request.samples))


@router.post("/tune", response_model=TuneResponse)
def tune_endpoint(request: GNNTuneRequest):
    request.config["vocab_size"] = request.vocab_size
    request.config["task"] = request.task
    
    result = tune(
        train_json=request.data.train,
        val_json=request.data.val,
        model_class=GNNModel,
        model_config=Config,
        config=Config(**request.config),
        experiment_name=request.experiment_name,
        n_trials=request.n_trials
    )
    
    return TuneResponse(
        best_params=result["best_params"],
        best_value=result["best_val_loss"],
        metric=result["metric"],
        trials=result["n_trials"]
    )


@router.post("/tune/stop")
def stop_tuning_endpoint():
    stop_tuning()
    return {"status": "tuning stopped"}


@router.get("/config", response_model=ConfigResponse)
def get_default_config():
    default_config = Config()
    return ConfigResponse(config=vars(default_config))
    