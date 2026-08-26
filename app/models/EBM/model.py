from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import mlflow
import mlflow.pyfunc
import joblib
import os

from app.models.ModelMixins.BaseModel import BaseModel

TASK_MAP = {
    "multiclass_classification": ExplainableBoostingClassifier,
    "classification": ExplainableBoostingClassifier,
    "regression": ExplainableBoostingRegressor,
}

class EBMModel(BaseModel):
    def __init__(self, config=None, task="classification", experiment_name="ebm", run_name="run", feature_names=None, run_id=None):
        super().__init__(config, task, experiment_name, run_name, feature_names, run_id)
        self.model_type = "ebm"
        self.model = TASK_MAP[task](feature_names=feature_names,**config.__dict__) if config is not None else None

    def _log_extra(self):
        self._save_global_explanation()

    def _save_model(self):
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EBMWrapper(self.model),
            conda_env=mlflow.pyfunc.get_default_conda_env()
        )

        # Save using joblib as well
        joblib.dump(self.model, "ebm.pkl")
        mlflow.log_artifact("ebm.pkl")
        os.remove("ebm.pkl")

    def load_from_run(self, run_id):
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, "ebm.pkl")
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "regression")
        self.model = joblib.load(local_path)
        self.run_id = run_id

    def _save_global_explanation(self):
        if self.model:
            self.ebm_importances = [self.model.term_importances(importance_type='avg_weight').tolist(), self.model.term_names_]
            explanation = self.model.explain_global()
            fig = explanation.visualize()
            fig.write_image("global_explanation.png")
            mlflow.log_artifact("global_explanation.png")
            os.remove("global_explanation.png")
        else:
            raise ValueError("Model is not loaded or trained yet.")

    def get_global_explanation(self):
        if self.model:
            return self.model.explain_global().visualize().to_json()
        else:
            raise ValueError("Model is not loaded or trained yet.")


class EBMWrapper(mlflow.pyfunc.PythonModel): 
    def __init__(self, model): 
        self.model = model 

    def predict(self, model_input): 
        return self.model.predict(model_input)
    