from fastapi import FastAPI, HTTPException, Request, status
from app.models.EBM.router import router as ebm_router
from app.models.GNN.router import router as gnn_router
from app.models.MLP.router import router as mlp_router
from app.models.RandomForest.router import router as random_forest_router
from app.models.Transformer.router import router as transformer_router
from app.models.XGBoost.router import router as xgboost_router
from app.models.ModelMixins.tune import stop_tuning
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn
import os

import torch
torch.set_float32_matmul_precision('medium')

import logging
import warnings

# Suppress torchmetrics buffer warning (SpearmanCorrCoef, etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

# Suppress Lightning pytree deprecation warning
warnings.filterwarnings("ignore", message=".*LeafSpec.*", category=UserWarning)

# Suppress Lightning num_workers bottleneck hint
warnings.filterwarnings("ignore", message=".*num_workers.*", category=UserWarning)

import mlflow
from mlflow.tracking import MlflowClient

from app.utils.model_registry import get_model_registry

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

api = FastAPI()
api.include_router(gnn_router)
api.include_router(ebm_router)
api.include_router(random_forest_router)
api.include_router(transformer_router)
api.include_router(xgboost_router)
api.include_router(mlp_router)


logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)

log = logging.getLogger("uvicorn.error")

@api.middleware("http")
async def log_request(request: Request, call_next):
    raw = await request.body()
    log.info(
        "REQ %s %s ct=%s cl=%s raw_len=%d",
        request.method,
        request.url.path,
        request.headers.get("content-type"),
        request.headers.get("content-length"),
        len(raw),
    )
    if raw:
        log.info("REQ body preview: %s", raw.decode("utf-8", errors="replace")[:1000])

    response = await call_next(request)
    log.info("RES %s %s status=%d", request.method, request.url.path, response.status_code)
    return response

@api.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw = await request.body()
    log.error("422 on %s %s", request.method, request.url.path)
    log.error("422 errors: %s", exc.errors())
    log.error("422 raw_len=%d raw_preview=%s", len(raw), raw.decode("utf-8", errors="replace")[:1000])
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@api.post("/stop_tuning", status_code=status.HTTP_202_ACCEPTED)
def stop_tuning_endpoint():
    stop_tuning()
    log.info("Tuning stopped by user request")
    return {"status": "tuning stopped"}

@api.get('/runs/{run_id}')
def get_run_info(run_id: str):
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        if run is None:
            return {"run_info": []}

        run_info = {
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "status": run.info.status,
        }

        return {"run_info": run_info}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get runs info.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@api.delete('/runs/{run_id}')
def delete_run(run_id: str):
    try:
        client = MlflowClient()
        client.delete_run(run_id)
        return JSONResponse(content=jsonable_encoder({"status": "success"}))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete run.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@api.delete('/experiments/{experiment_id}')
def delete_experiment(experiment_id: str):
    try:
        client = MlflowClient()
        client.delete_experiment(experiment_id)
        return JSONResponse(content=jsonable_encoder({"status": "success"}))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete experiment.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

@api.get('/model/registry')
def model_registry():
    try:
        return JSONResponse(content=jsonable_encoder({"models": list(get_model_registry().keys())}))
    except Exception as e:
        logger.exception("Failed to get model registry.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)
    
@api.get('/hello')
def hello():
    try:
        return JSONResponse(content=jsonable_encoder({"message": "Hello, world!"}))
    except Exception as e:
        logger.exception("Hello endpoint failed.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

if __name__ == '__main__':
    api.run(debug=True)
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")
