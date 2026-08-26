from xgboost import XGBClassifier, XGBRegressor
import mlflow

from app.models.ModelMixins.BaseModel import BaseModel

TASK_MAP = {
    "multiclass_classification": XGBClassifier,
    "classification": XGBClassifier,
    "regression": XGBRegressor,
}

class XGBoostModel(BaseModel):
    def __init__(self, config=None, task="regression", experiment_name="xgboost", run_name="run", feature_names=None, run_id=None):
        super().__init__(config, task, experiment_name, run_name, feature_names, run_id)
        self.model_type = "xgboost"
        self.model = TASK_MAP[task](**config.__dict__) if config is not None else None
    
    def _save_model(self):
        mlflow.xgboost.log_model(self.model, name="model")

    def load_from_run(self, run_id):
        self.model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        self.run_id = run_id
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "regression")
