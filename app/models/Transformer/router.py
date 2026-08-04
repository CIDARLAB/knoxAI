import logging

from app.utils.schemas import *
from app.models.Transformer.config import Config
from app.models.Transformer.model import TransformerModel
from app.utils.routers import create_model_router_pytorch

logging.basicConfig( level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" )
logger = logging.getLogger(__name__)
log = logging.getLogger("uvicorn.error")

router = create_model_router_pytorch(
    model_cls = TransformerModel,
    config_cls = Config,
    prefix="/transformer",
    tags=["Transformer"],
    train_request_type = TransformerTrainRequest,
    predict_request_type = TransformerPredictRequest,
    tune_request_type = TransformerTuneRequest,
)
