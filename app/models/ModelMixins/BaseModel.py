import mlflow
import pandas as pd
import numpy as np
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import sklearn.metrics as metric
from sklearn.model_selection import cross_val_score
import shap
import scipy.stats as stat

from app.models.ModelMixins.data_module import PytorchDataModule, log_dataset
from app.models.ModelMixins.run_shap import plot_shap_top10

class ModelMixin:
    def __init__(self, config, model_type, task="regression"):
        self.config = config
        self.task = task
        self.model_type = model_type
    
    def _log_metrics(self, y_test, y_pred, y_pred_proba=None):
        #print(f"Shape of y_test: {y_test.shape}, Shape of y_pred: {y_pred.shape}")

        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test = np.ravel(y_test)

        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        metrics = {}
        if self.task == "regression":
            metrics['test_R2'] = metric.r2_score(y_test, y_pred)
            metrics['test_MSE'] = metric.mean_squared_error(y_test, y_pred)
            metrics['test_RMSE'] = metric.root_mean_squared_error(y_test, y_pred)
            metrics['test_MAE'] = metric.mean_absolute_error(y_test, y_pred)
            metrics['test_max_error'] = metric.max_error(y_test, y_pred)
            metrics['test_Kendall'] = stat.kendalltau(y_test, y_pred).statistic
            metrics['test_Spearman'] = stat.spearmanr(y_test, y_pred).statistic
            metrics['test_Pearson'] = stat.pearsonr(y_test, y_pred).statistic

        elif self.task == "classification":
            metrics["test_accuracy"] = metric.accuracy_score(y_test, y_pred)
            metrics["test_f1"] = metric.f1_score(y_test, y_pred, average="weighted")
            metrics["test_precision"] = metric.precision_score(y_test, y_pred, average="weighted")
            metrics["test_recall"] = metric.recall_score(y_test, y_pred, average="weighted")
            metrics["test_roc_auc"] = metric.roc_auc_score(y_test, y_pred, average="weighted")

        elif self.task == "multiclass_classification":
            metrics["test_accuracy"] = metric.accuracy_score(y_test, y_pred)
            metrics["test_f1"] = metric.f1_score(y_test, y_pred, average="macro")
            metrics["test_precision"] = metric.precision_score(y_test, y_pred, average="macro")
            metrics["test_recall"] = metric.recall_score(y_test, y_pred, average="macro")
            metrics["test_roc_auc"] = metric.roc_auc_score(y_test, y_pred_proba, average="macro", multi_class="ovo")
            
        mlflow.log_metrics(metrics)

    def _log_params(self):
        mlflow.log_params(self.config.__dict__)
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("task", self.task)

    
class BaseModel(ModelMixin):
    def __init__(self, config=None, task="regression", experiment_name="default_experiment", run_name="run", feature_names=None, run_id=None):
        super().__init__(config, model_type="base_model", task=task)
        self.feature_names = feature_names
        self.model = None
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        self.shap_values = None
        self.ebm_importances = None

        self.score_map = {
            "regression": "R2",
            "classification": "accuracy",
            "multiclass_classification": "accuracy"
        }

    def train(self, x, y, save_model=True):
        run_ctx = mlflow.start_run(run_id=self.run_id) if self.run_id else mlflow.start_run(run_name=self.run_name)
        with run_ctx as run:
            # Load data to numpy arrays
            X_train, y_train = self._load_data(x, y)

            # Handle Feature names
            self.feature_names = self.feature_names or range(X_train.shape[1])

            # Log dataset
            self._log_dataset(X_train, y_train, context="training")

            # Log hyperparameters
            self._log_params()

            # Train the model
            self.model.fit(X_train, y_train)

            # Log training metrics
            mlflow.log_metric(f"train_{self.score_map[self.task]}", self.model.score(X_train, y_train))

            # Log Extra Information
            self._log_extra()

            # Log the model itself
            self._save_model() if save_model else None

            # Store run_id for future reference
            self.run_id = run.info.run_id

    def cross_validate(self, x, y, cv=5, scoring="neg_mean_squared_error"):
        with mlflow.start_run(run_name=self.run_name):
            # Load data to numpy arrays
            x, y = self._load_data(x, y)
            
            # Log dataset
            self._log_dataset(x, y, context="tuning")

            # Log hyperparameters
            self._log_params()

            # Run cross-validation
            scores = -cross_val_score(self.model, x, y, cv=cv, scoring=scoring)

            # Log CV metrics
            mlflow.log_metric("val_mse_cv", scores.mean())

            # Store run_id for future reference
            self.run_id = mlflow.active_run().info.run_id

        return scores

    def evaluate(self, X_test, y_test):
        with mlflow.start_run(run_id=self.run_id):
            # Load data to numpy arrays
            X_test, y_test = self._load_data(X_test, y_test)

            # Log test dataset
            self._log_dataset(X_test, y_test, context="testing")

            # Get predictions
            y_pred = self.predict(X_test)
            y_pred_proba = None
            if self.task == "multiclass_classification":
                y_pred_proba = self.predict_proba(X_test)

            # Log metrics
            self._log_metrics(y_test, y_pred, y_pred_proba=y_pred_proba)

    def predict(self, X, sample_ids=None, save_predictions=False):
        X = np.array(X)
        preds = self.model.predict(X)

        if save_predictions:
            with mlflow.start_run(run_id=self.run_id):
                df = pd.DataFrame()
                if sample_ids is not None:
                    df["sample_id"] = sample_ids
                df["prediction"] = preds
                mlflow.log_table(data=df, artifact_file="predictions.json")

        return preds

    def predict_proba(self, X, sample_ids=None, save_predictions=False):
        X = np.array(X)
        preds_proba = self.model.predict_proba(X)

        if save_predictions:
            with mlflow.start_run(run_id=self.run_id):
                df = pd.DataFrame()
                if sample_ids is not None:
                    df["sample_id"] = sample_ids
                df["prediction"] = preds_proba
                mlflow.log_table(data=df, artifact_file="predictions.json")

        return preds_proba
    
    def interpret_shap(self, X_train, X_test):
        if self.model_type == "ebm":
            return 
        
        with mlflow.start_run(run_id=self.run_id):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
            X_test = pd.DataFrame(X_test, columns=self.feature_names)

            background = shap.sample(X_train, 500, random_state=42)
            explainer = shap.TreeExplainer(self.model, data=background, feature_perturbation="interventional")
            shap_values = explainer(X_test)
            self.shap_values = shap_values
            plot_shap_top10(X_test, self.shap_values.values, self.feature_names, self.model_type)
    
    def _load_data(self, X, y):
        return np.array(X), np.array(y)

    def _log_dataset(self, X, y, context="training"):
        df = pd.DataFrame(X, columns=self.feature_names)
        df["target"] = y
        dataset = mlflow.data.from_pandas(
            df=df.astype("float64"),
            targets="target"
        )
        
        mlflow.log_input(dataset, context=context)

    def _save_model(self):
        # Override in subclasses to save the model appropriately
        pass

    def _log_extra(self):
        # Extra Logging specific to the model, Used only during training
        # Implement in subclasses if needed
        pass


class PytorchBaseModel(ModelMixin):
    def __init__(self, config=None, task="regression", experiment_name="default_experiment", run_name="run", vocab=None, run_id=None):
        super().__init__(config, model_type="pytorch_base_model", task=task)
        self.vocab = vocab
        self.model = None
        self.backbone = None
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        self.shap_values = None

    def train(self, train_json, val_json, test_json=None, save_model=False, patience=20, min_delta=1e-4):
        run_ctx = mlflow.start_run(run_id=self.run_id) if self.run_id else mlflow.start_run(run_name=self.run_name)
        with run_ctx as run:
            seed_everything(self.config.seed)
            
            # -----------------------------
            # MLflow setup
            # -----------------------------
            self.run_id = run.info.run_id
            mlf_logger = MLFlowLogger(
                experiment_name=self.experiment_name,
                run_id=self.run_id,
                tracking_uri=mlflow.get_tracking_uri() 
            )

            self._log_dataset("training", train_json)
            self._log_dataset("val", val_json)
            self._log_dataset("testing", test_json) if test_json is not None else None
            mlflow.log_param("num_trainable_params", self._count_trainable_params())
            mlflow.log_param("model_size_MB", self._model_size_bytes() / 1024 / 1024)
            self._log_params()

            # -----------------------------
            # Data
            # -----------------------------
            dm = self._load_data_module(train_json, val_json, test_json)

            # -----------------------------
            # Callbacks
            # -----------------------------
            ckpt_cb, early_stop = self._check_points(save_model, patience=patience, min_delta=min_delta)

            # -----------------------------
            # Trainer
            # -----------------------------
            trainer = Trainer(
                max_epochs=self.config.max_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                logger=mlf_logger,
                callbacks=[ckpt_cb, early_stop],
                log_every_n_steps=10
            )

            # -----------------------------
            # Train
            # -----------------------------
            trainer.fit(self.model, datamodule=dm)

            best_ckpt_path = ckpt_cb.best_model_path
            if best_ckpt_path and (test_json or save_model):
                self._load_from_checkpoint(best_ckpt_path)

            # Log best val loss as a summary metric
            mlflow.log_metric("best_val_loss", early_stop.best_score.item())

            # -----------------------------
            # Test on best checkpoint
            # -----------------------------
            if test_json is not None:
                trainer.test(self.model, datamodule=dm, ckpt_path=ckpt_cb.best_model_path)
                self.evaluate(test_json)

            # -----------------------------
            # Log final model to MLflow
            # -----------------------------
            self._save_model() if save_model else None

    def _count_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _model_size_bytes(self):
        return sum(p.element_size() * p.numel() for p in self.model.parameters())

    def _load_from_checkpoint(self, ckpt_path):
        self.model = self.model.__class__.load_from_checkpoint(ckpt_path)
        self.backbone = self.model.backbone

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.backbone.to(device)

    def evaluate(self, test_json):
        # Log test dataset
        self._log_dataset("testing", test_json)

        # Get predictions
        y_pred = self.predict(test_json)
        y_pred_proba = None
        if self.task == "multiclass_classification":
            y_pred_proba = self.predict_proba(test_json)

        # Log metrics
        y_true = np.array([sample.y for sample in test_json])
        self._log_metrics(np.array(y_true), np.array(y_pred), y_pred_proba=np.array(y_pred_proba) if y_pred_proba is not None else None)

    def predict_logits(self, samples):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(device)
        self.backbone.eval()
        pyg_samples = self._convert_dataset(samples)

        collate_fn = self._get_collate_fn()
        if collate_fn is None:
            loader = PyGDataLoader(pyg_samples, batch_size=len(pyg_samples))
        else:
            loader = TorchDataLoader(
                pyg_samples,
                batch_size=len(pyg_samples),
                collate_fn=collate_fn
            )

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = self._predict_helper(batch)

        return logits.cpu()

    def predict_proba(self, samples):
        logits = self.predict_logits(samples)

        if self.task == "classification":
            return torch.sigmoid(logits).cpu().numpy().tolist()

        if self.task == "multiclass_classification":
            return torch.softmax(logits, dim=-1).cpu().numpy().tolist()

        raise ValueError("predict_proba is only supported for classification tasks")

    def predict(self, samples, sample_ids=None, save_predictions=False):
        logits = self.predict_logits(samples)

        if self.task == "regression":
            self.save_predictions(logits.cpu().numpy().tolist(), sample_ids=sample_ids) if save_predictions else None
            return logits.cpu().numpy().tolist()

        if self.task == "classification":
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            self.save_predictions(preds.cpu().numpy().tolist(), sample_ids=sample_ids) if save_predictions else None
            return preds.cpu().numpy().tolist()

        if self.task == "multiclass_classification":
            preds = torch.argmax(logits, dim=-1)
            self.save_predictions(preds.cpu().numpy().tolist(), sample_ids=sample_ids) if save_predictions else None
            return preds.cpu().numpy().tolist()

        return logits.cpu().numpy().tolist()

    def save_predictions(self, preds, sample_ids=None):
        with mlflow.start_run(run_id=self.run_id):
            df = pd.DataFrame()
            if sample_ids is not None:
                df["sample_id"] = sample_ids
            df["prediction"] = preds
            mlflow.log_table(data=df, artifact_file="predictions.json")
    
    def build_surrogate_model(
            self, 
            model_class,
            model_config,
            train_rule_matrix,
            test_rule_matrix,
            train_json, 
            val_json,
            test_json=None,
            rule_names=None,
        ):

        # Get predictions from Transformer backbone
        self.backbone.eval()
        preds = self.predict(train_json)
        preds += self.predict(val_json)

        # Initialize surrogate model
        self.surrogate_model = model_class(
            model_config, 
            task=self.task, 
            experiment_name=self.experiment_name,
            run_name=f"{self.run_name}_surrogate",
            feature_names=rule_names
        )

        # Train surrogate model
        self.surrogate_model.train(
            train_rule_matrix,
            preds,
            save_model=True
        )

        # Evaluate surrogate model if test data is provided
        if test_json is not None:
            test_preds = self.predict(test_json)
            self.surrogate_model.evaluate(
                test_rule_matrix,
                test_preds
            )

            self.surrogate_model.interpret_shap(
                train_rule_matrix,
                test_rule_matrix
            )

            self.shap_values = self.surrogate_model.shap_values

            return self.surrogate_model.shap_values.values.tolist()

    def _check_points(self, save_model, patience=20, min_delta=1e-4):
        ckpt_dir = f"checkpoints/{self.run_id}"

        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1 if save_model else 0,
            filename="best-{epoch}-{val_loss:.4f}"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            min_delta=min_delta
        )

        return ckpt_cb, early_stop
    
    def _save_model(self):
        mlflow.pytorch.log_model(
            pytorch_model=self.backbone,
            name="model"
        )
    
    def load_inference_model(self, run_id):
        self.run_id = run_id
        run = mlflow.get_run(run_id)
        self.task = run.data.params.get("task", "regression")
        model_uri = f"runs:/{run_id}/model"
        self.backbone = mlflow.pytorch.load_model(model_uri)
        self.backbone.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(device)

    def _load_data_module(self, train_json, val_json, test_json=None):
        dm = PytorchDataModule(
            train_data=self._convert_dataset(train_json),
            val_data=self._convert_dataset(val_json),
            test_data=self._convert_dataset(test_json) if test_json is not None else None,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self._get_collate_fn()
        )
        return dm
    
    def _log_dataset(self, context, json_list):
        log_dataset(context, json_list)
    
    def _convert_dataset(self, json_list):
        return [self._convert_json_to_pyg(sample) for sample in json_list]
    
    def _convert_json_to_pyg(self, sample):
        # Set by Specific Model Implementations
        pass

    def _predict_helper(self, batch):
        # To be implemented in specific model implementations
        pass

    def _get_collate_fn(self):
        # Override in models that need custom padding/masking
        return None

        