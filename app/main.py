from fastapi import FastAPI, HTTPException
from app.models.GNN.router import router as gnn_router
from app.models.EBM.router import router as ebm_router
from app.models.XGBoost.router import router as xgboost_router
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging

from mlflow.tracking import MlflowClient

from app.utils.model_registry import get_model_registry
from app.utils.metrics import get_available_metrics

api = FastAPI()
api.include_router(gnn_router)
api.include_router(ebm_router)
api.include_router(xgboost_router)

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)

@api.get('/runs')
def get_runs_info(experiment_name: str = "ebm"):
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return {"run_info": []}

        runs = client.search_runs([experiment.experiment_id])
        run_info = [ 
            { 
                "run_id": run.info.run_id, 
                "start_time": run.info.start_time, 
                "end_time": run.info.end_time, 
                "metrics": run.data.metrics, 
                "params": run.data.params, 
            } for run in runs ]

        return {"run_info": run_info}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get runs info.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)


@api.get('/available-metrics')
def available_metrics():
    try:
        return JSONResponse(content=jsonable_encoder(get_available_metrics().keys()))
    except Exception as e:
        logger.exception("Failed to get available metrics.")
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
