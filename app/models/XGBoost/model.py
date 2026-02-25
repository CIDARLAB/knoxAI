from xgboost import XGBClassifier, XGBRegressor
import mlflow
import sklearn.metrics as metric
import scipy.stats as stat
import pandas as pd

TASK_MAP = {
    "classification": XGBClassifier,
    "regression": XGBRegressor,
}

SCORE_MAP = {
    "classification": "accuracy",
    "regression": "r2_score",
}

class XGBoostModel:
    def __init__(self, config=None, task="classification", experiment_name="xgboost"):
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
            mlflow.xgboost.log_model(self.model, name="model")

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
        self.model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        self.run_id = run_id
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "classification")
