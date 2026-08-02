from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import joblib
import os
import sklearn.metrics as metric
import scipy.stats as stat
import pandas as pd

from app.models.ModelMixins.BaseModel import BaseModel

TASK_MAP = {
    "classification": RandomForestClassifier,
    "multiclass_classification": RandomForestClassifier,
    "regression": RandomForestRegressor,
}

class RandomForestModel(BaseModel):
    def __init__(self, config=None, task="regression", experiment_name="random_forest", run_name="run", feature_names=None):
        super().__init__(config, task, experiment_name, run_name, feature_names)
        self.model_type = "random_forest"
        self.model = TASK_MAP[task](**config.__dict__)

    def _log_extra(self):
        # Log Feature importances
        mlflow.log_dict(
            {"importances": dict(zip(self.feature_names, self.model.feature_importances_))},
            "feature_importances.json"
        )
    
    def _save_model(self):
        mlflow.sklearn.log_model(self.model, name="model")
    
    def load_from_run(self, run_id):
        self.run_id = run_id
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "regression")
        self.model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

