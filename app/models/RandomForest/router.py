import logging
logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)

from app.utils.routers import create_model_router
from app.utils.schemas import *
from app.models.RandomForest.model import RandomForestModel
from app.models.RandomForest.config import Config
from app.models.RandomForest.tune import tune, stop_tuning

router = create_model_router(
    model_cls=RandomForestModel,
    config_cls=Config,
    prefix="/random_forest",
    tags=["Random Forest"],
    tune_fn=tune,
    stop_tune_fn=stop_tuning
)
    