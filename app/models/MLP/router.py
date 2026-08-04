import logging

from app.utils.schemas import *
from app.models.MLP.config import Config
from app.models.MLP.model import MLPModel
from app.utils.routers import create_model_router_pytorch

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)
log = logging.getLogger("uvicorn.error")

router = create_model_router_pytorch(
    model_cls = MLPModel,
    config_cls = Config,
    prefix="/mlp",
    tags=["MLP"],
    train_request_type = MLPTrainRequest,
    predict_request_type = MLPPredictRequest,
    tune_request_type = MLPTuneRequest,
)
