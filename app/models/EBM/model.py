from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import mlflow
import mlflow.pyfunc
import joblib
import os
import sklearn.metrics as metric
import scipy.stats as stat
import pandas as pd

TASK_MAP = {
    "classification": ExplainableBoostingClassifier,
    "regression": ExplainableBoostingRegressor,
}

SCORE_MAP = {
    "classification": "accuracy",
    "regression": "r2_score",
}

class EBMModel:
    def __init__(self, config=None, task="classification", experiment_name="ebm"):
        self.config = config
        self.task = task
        self.model = None

        if config:
            self.model = TASK_MAP[task](**config)

        mlflow.set_experiment(experiment_name)

    def train(self, X_train, y_train, feature_names=None):
        with mlflow.start_run():
            # Log dataset
            if feature_names:
                df = pd.DataFrame(X_train, columns=feature_names)
            else:
                df = pd.DataFrame(X_train)
            df["target"] = y_train
            dataset = mlflow.data.from_pandas(
                df=df.astype("float64"),
                targets="target"
            )
            mlflow.log_input(dataset, context="training")

            # Log hyperparameters
            mlflow.log_params(self.config)
            mlflow.log_param("task", self.task)

            self.model.fit(X_train, y_train)

            # Log training metrics
            mlflow.log_metric(f"train_{SCORE_MAP[self.task]}", self.model.score(X_train, y_train))

            # Log the model itself
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=EBMWrapper(self.model),
                conda_env=mlflow.pyfunc.get_default_conda_env()
            )

            # Save Global Explanation
            self.save_global_explanation()

            # Save using joblib as well
            joblib.dump(self.model, "ebm.pkl")
            mlflow.log_artifact("ebm.pkl")
            os.remove("ebm.pkl")

            self.run_id = mlflow.active_run().info.run_id

    def evaluate(self, X_test, y_test, feature_names=None):
        # Log test dataset
        if feature_names:
            df = pd.DataFrame(X_test, columns=feature_names)
        else:
            df = pd.DataFrame(X_test)
        df["target"] = y_test
        dataset = mlflow.data.from_pandas(df.astype("float64"), targets="target")
        

        preds = self.predict(X_test)
        metrics = {}

        if self.task == "classification":
            metrics = {
                "test_accuracy": metric.accuracy_score(y_test, preds),
                "test_f1": metric.f1_score(y_test, preds, average="weighted")
            }
        else:
            metrics['test_r2_score'] = metric.r2_score(y_test, preds)
            metrics['test_mse'] = metric.mean_squared_error(y_test, preds)
            metrics['test_rmse'] = metric.root_mean_squared_error(y_test, preds)
            metrics['test_mae'] = metric.mean_absolute_error(y_test, preds)
            metrics['test_kendalls'] = stat.kendalltau(y_test, preds).statistic
            metrics['test_spearmans'] = stat.spearmanr(y_test, preds).statistic
            metrics['test_pearsons'] = stat.pearsonr(y_test, preds).statistic

        # Log to the same run if active, otherwise start a new one
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_input(dataset, context="testing")
            mlflow.log_metrics(metrics)

        return metrics

    def predict(self, X):
        return self.model.predict(X)

    def load_from_run(self, run_id):
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, "ebm.pkl")
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "classification")
        self.model = joblib.load(local_path)
        self.run_id = run_id

    def save_global_explanation(self):
        if self.model:
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
    