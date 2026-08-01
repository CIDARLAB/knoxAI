import optuna
import threading
from app.models.RandomForest.config import Config
from app.models.RandomForest.model import RandomForestModel

_stop_event = threading.Event()

def stop_tuning():
    _stop_event.set()

def tune(x_train, y_train, task="regression", base_config=None, feature_names=None, experiment_name="random_forest_tuning", n_trials=30):
    _stop_event.clear()

    if base_config is None:
        base_config = Config()

    def objective(trial):
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", [0.15, "sqrt", "log2"]),
        }

        config = Config(**{**vars(base_config), **trial_params})

        model = RandomForestModel(
            config=config, 
            task=task, 
            experiment_name=experiment_name, 
            run_name=f"trial_{trial.number}",
            feature_names=feature_names
        )

        return model.cross_validate(x_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    
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
        "best_value": best.value
    }
