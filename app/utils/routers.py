from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import logging

from app.utils.schemas import *
from app.utils.mlflow_utils import create_train_run
from app.models.ModelMixins.tune import tune_pytorch, tune
from app.models.ModelMixins.BaseModel import BaseModel, PytorchBaseModel
from app.models.RandomForest.model import RandomForestModel
from app.models.RandomForest.config import Config as RFConfig

def create_model_router(
    *,
    model_cls : BaseModel,
    config_cls,
    prefix: str,
    tags: list
):
    router = APIRouter(prefix=prefix, tags=tags)
    logger = logging.getLogger(__name__)
    log = logging.getLogger("uvicorn.error")


    @router.post("/train", response_model=TrainResponse)
    def train_endpoint(request: TrainRequest):
        try:
            # Prepare config
            config = config_cls(**request.config) if request.config else config_cls()

            # Initialize model
            model = model_cls(
                config=config, 
                experiment_name=request.experiment_name, 
                task=request.task, 
                run_name=request.run_name,
                feature_names=request.feature_names
            )

            # Train the model
            model.train(
                request.data.x_train, 
                request.data.y_train
            )

            # Evaluate the model
            if request.data.x_test is not None and request.data.y_test is not None:
                model.evaluate(
                    request.data.x_test, 
                    request.data.y_test
                )

                # Interpret with SHAP if requested
                if request.interpret_shap:
                    model.interpret_shap(
                        request.data.x_train,
                        request.data.x_test
                    )

            return TrainResponse(run_id=model.run_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Training failed.")
            return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)


    @router.post("/predict", response_model=PredictResponse)
    def predict_endpoint(request: PredictRequest):
        try:
            # Initialize model
            model = model_cls()

            # Load model from run_id
            model.load_from_run(request.run_id)

            # Return predictions
            return PredictResponse(predictions=model.predict(request.samples).tolist())
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Prediction failed.")
            return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)


    @router.post("/evaluate", response_model=EvaluateResponse)
    def evaluate_endpoint(request: EvaluateRequest):
        try:
            # Initialize model
            model = model_cls(feature_names=request.feature_names)

            # Load model from run_id
            model.load_from_run(request.run_id)

            # Evaluate the model
            metrics = model.evaluate(request.x_test, request.y_test)

            # Return evaluation metrics
            return EvaluateResponse(metrics=metrics)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Evaluation failed.")
            return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)


    @router.post("/tune", status_code=status.HTTP_202_ACCEPTED)
    async def tune_endpoint(request_raw: Request, request: TuneRequest, background_tasks : BackgroundTasks):
        raw = await request_raw.body()
        log.info("raw_len=%d", len(raw))
        background_tasks.add_task(run_tune, request.model_dump())

    def run_tune(payload: dict) -> None:
        request = TuneRequest(**payload)
        tune(
            x_train=request.data.x_train,
            y_train=request.data.y_train,
            task=request.task,
            model_class=model_cls,
            model_config=config_cls,
            config=config_cls(**request.config) if request.config else config_cls(),
            feature_names=request.feature_names,
            experiment_name=request.experiment_name,
            n_trials=request.n_trials
        )
        log.info("Tuning completed")


    @router.get("/config", response_model=ConfigResponse)
    def config_endpoint():
        try:
            default_config = config_cls()
            return ConfigResponse(config=vars(default_config))
        except Exception as e:
            logger.exception("Failed to get default config.")
            return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)

    return router


def create_model_router_pytorch(
    *,
    model_cls : PytorchBaseModel,
    config_cls,
    prefix: str,
    tags: list,
    train_request_type : type,
    predict_request_type : type,
    tune_request_type : type,
):
    router = APIRouter(prefix=prefix, tags=tags)
    logger = logging.getLogger(__name__)
    log = logging.getLogger("uvicorn.error")

    @router.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
    async def train_endpoint(
        request_raw: Request, 
        request: train_request_type,
        background_tasks : BackgroundTasks
    ):
        raw = await request_raw.body()
        log.info("raw_len=%d", len(raw))

        run_id = create_train_run(request.experiment_name, request.run_name)
        background_tasks.add_task(run_training, request.model_dump(), run_id)

        return TrainResponse(run_id=run_id)

    def run_training(payload: dict, run_id: str) -> None:
        request = train_request_type(**payload)

        request.config["vocab_size"] = request.vocab_size + 1
        request.config["task"] = request.task

        model = model_cls(
            config=config_cls(**request.config),
            task=request.task,
            experiment_name=request.experiment_name,
            run_name=request.run_name,
        )

        model.train(
            train_json=request.data.train,
            val_json=request.data.val,
            test_json=request.data.test,
            save_model=True,
            run_id=run_id,
        )

        if request.rule_matrix is not None:
            model.build_surrogate_model(
                model_class=RandomForestModel,
                model_config=RFConfig(),
                train_rule_matrix=request.rule_matrix.x_train,
                test_rule_matrix=request.rule_matrix.x_test,
                train_json=request.data.train,
                val_json=request.data.val,
                test_json=request.data.test,
                rule_names=request.rule_matrix.feature_names,
            )

    @router.post("/predict", response_model=PredictResponse)
    def predict_endpoint(request: predict_request_type):
        model = model_cls(config=config_cls(**request.config))

        model.load_inference_model(request.run_id)

        return PredictResponse(predictions=model.predict(request.samples))

    @router.post("/tune", status_code=status.HTTP_202_ACCEPTED)
    async def tune_endpoint(request: tune_request_type, request_raw: Request, background_tasks: BackgroundTasks):
        raw = await request_raw.body()
        log.info("raw_len=%d", len(raw))
        background_tasks.add_task(run_tune, request.model_dump())

    def run_tune(payload: dict) -> None:
        request = tune_request_type(**payload)
        request.config["vocab_size"] = request.vocab_size + 1
        request.config["task"] = request.task

        tune_pytorch(
            train_json=request.data.train,
            val_json=request.data.val,
            task=request.task,
            model_class=model_cls,
            model_config=config_cls,
            config=config_cls(**request.config) if request.config else config_cls(),
            experiment_name=request.experiment_name,
            n_trials=request.n_trials
        )

        log.info("Tuning completed")


    @router.get("/config", response_model=ConfigResponse)
    def get_default_config():
        default_config = config_cls()
        return ConfigResponse(config=vars(default_config))

    return router
