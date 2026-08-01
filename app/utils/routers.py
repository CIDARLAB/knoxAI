from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import logging
import numpy as np

from app.utils.schemas import *

def create_model_router(
    *,
    model_cls,
    config_cls,
    prefix: str,
    tags: list,
    tune_fn=None,
    stop_tune_fn=None
):
    router = APIRouter(prefix=prefix, tags=tags)
    logger = logging.getLogger(__name__)


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

            return TrainResponse(
                run_id=model.run_id, 
                shap_values=model.shap_values.values.tolist() if model.shap_values is not None else None, 
                ebm_importances=model.ebm_importances
            )
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


    @router.get("/config", response_model=ConfigResponse)
    def config_endpoint():
        try:
            default_config = config_cls()
            return ConfigResponse(config=vars(default_config))
        except Exception as e:
            logger.exception("Failed to get default config.")
            return JSONResponse(content=jsonable_encoder({"error": "Internal server error occurred"}), status_code=500)


    if tune_fn is not None and stop_tune_fn is not None:
        @router.post("/tune")
        def tune_endpoint(request: TuneRequest):
            return tune_fn(
                x_train=request.data.x_train,
                y_train=request.data.y_train,
                base_config=config_cls(**request.config) if request.config else config_cls(),
                feature_names=request.feature_names,
                experiment_name=request.experiment_name,
                n_trials=request.n_trials
            )


        @router.post("/tune/stop")
        def stop_tuning_endpoint():
            stop_tune_fn()
            return {"status": "tuning stopped"}

    return router