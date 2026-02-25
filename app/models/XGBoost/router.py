from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal

import logging

import numpy as np

from .model import XGBoostModel
from .config import XGBOOST_CONFIG, EXPERIMENT_NAME

router = APIRouter(prefix="/xgboost", tags=["XGBoost"])

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)


class XGBTrainRequest(BaseModel):
    data: dict
    feature_names: list[str] | None = None
    task: Literal["classification", "regression"] = "classification"
    config: dict = Field(default_factory=lambda: XGBOOST_CONFIG.copy())
    experiment_name: str = EXPERIMENT_NAME

class TrainResponse(BaseModel):
    run_id: str
    task: str

@router.post("/train", response_model=TrainResponse)
def train(request: XGBTrainRequest):
    try:
        config = {**XGBOOST_CONFIG, **request.config}
        config["feature_names"] = request.feature_names
        model = XGBoostModel(config=config, experiment_name=request.experiment_name, task=request.task)
        model.train(np.array(request.data["X_train"]), np.array(request.data["y_train"]), feature_names=request.feature_names)
        return TrainResponse(run_id=model.run_id, task=request.task)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Training failed.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

class PredictRequest(BaseModel):
    run_id: str
    X_test: list
    y_test: list = None  # Optional, only for evaluation
    feature_names: list[str] | None = None # Optional, only for evaluation

@router.post("/predict")
def predict(request: PredictRequest):
    try:
        model = XGBoostModel()
        model.load_from_run(request.run_id)
        return JSONResponse(content=jsonable_encoder({"predictions": model.predict(np.array(request.X_test)).tolist()}))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@router.post("/evaluate")
def evaluate(request: PredictRequest):
    try:
        model = XGBoostModel()
        model.load_from_run(request.run_id)
        metrics = model.evaluate(np.array(request.X_test), np.array(request.y_test), feature_names=request.feature_names)
        return JSONResponse(content=jsonable_encoder({"metrics": metrics}))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Evaluation failed.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)
    
@router.get("/config")
def get_default_config():
    try:
        return JSONResponse(content=jsonable_encoder(XGBOOST_CONFIG))
    except Exception as e:
        logger.exception("Failed to get xgboost default config.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

