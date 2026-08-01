import optuna
import mlflow
import threading
from app.models.Transformer.config import Config
from app.models.Transformer.model import TransformerModel

_stop_event = threading.Event()

def stop_tuning():
    _stop_event.set()

def tune(train_json, val_json, base_config=None, experiment_name="transformer-tuning", n_trials=30):
    _stop_event.clear()

    if base_config is None:
        base_config = Config()

    trial_base = Config(**{**vars(base_config), "max_epochs": 150})

    def objective(trial):
        trial_params = {
            "model_dim"   : trial.suggest_categorical("model_dim", [64, 128, 256]),
            "num_heads"   : trial.suggest_categorical("num_heads", [4, 8]),
            "num_layers"  : trial.suggest_int("num_layers", 2, 4),
            "dropout"     : trial.suggest_float("dropout", 0.1, 0.4),
            "lr"          : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size"  : trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

        config = Config(**{**vars(trial_base), **trial_params})

        model = TransformerModel(
            config=config, 
            experiment_name=experiment_name, 
            run_name=f"trial_{trial.number}"
        )

        model.train(
            train_json=train_json,
            val_json=val_json,
            save_model=False
        )
        
        # Return val metric to minimise — pull from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model.run_id)
        return run.data.metrics["best_val_loss"]  # or val_loss
    
    def stop_callback(study, trial):
        if _stop_event.is_set():
            study.stop()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_callback])

    if not study.trials:
        return {"status": "stopped before any trials completed"}

    best = study.best_trial
    return {
        "best_params": best.params,
        "best_val_loss": best.value,
        "metric": "mse",
        "best_trial_number": best.number,
        "n_trials": len(study.trials),
    }
