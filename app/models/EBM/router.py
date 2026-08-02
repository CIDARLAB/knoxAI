from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)

from app.utils.routers import create_model_router
import numpy as np
from app.utils.schemas import *
from app.models.EBM.model import EBMModel
from app.models.EBM.config import Config
from app.models.EBM.tune import tune, stop_tuning

router = create_model_router(
    model_cls=EBMModel,
    config_cls=Config,
    prefix="/ebm",
    tags=["EBM"],
    tune_fn=tune,
    stop_tune_fn=stop_tuning
)
    
@router.get("/global_plot/{run_id}")
def get_global_plot(run_id: str):
    try:
        model = EBMModel()
        model.load_from_run(run_id)
        return {"plot": model.get_global_explanation()}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get global plot.")
        return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)
    