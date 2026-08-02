import logging

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)
from app.utils.routers import create_model_router
from app.utils.schemas import *
from app.models.XGBoost.model import XGBoostModel
from app.models.XGBoost.config import Config
from app.models.XGBoost.tune import tune, stop_tuning

router = create_model_router(
    model_cls=XGBoostModel,
    config_cls=Config,
    prefix="/xgboost",
    tags=["XGBoost"],
    tune_fn=tune,
    stop_tune_fn=stop_tuning
)
