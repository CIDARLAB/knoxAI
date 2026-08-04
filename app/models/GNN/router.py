import logging

from app.utils.schemas import *
from app.models.GNN.config import Config
from app.models.GNN.model import GNNModel
from app.utils.routers import create_model_router_pytorch

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)
log = logging.getLogger("uvicorn.error")

router = create_model_router_pytorch(
    model_cls = GNNModel,
    config_cls = Config,
    prefix="/gnn",
    tags=["GNN"],
    train_request_type = GNNTrainRequest,
    predict_request_type = GNNPredictRequest,
    tune_request_type = GNNTuneRequest,
)
    