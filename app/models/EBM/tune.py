import optuna
import threading
from app.models.EBM.config import Config
from app.models.EBM.model import EBMModel

_stop_event = threading.Event()

def stop_tuning():
    _stop_event.set()

def tune(x_train, y_train, task="regression", base_config=None, feature_names=None, experiment_name="ebm_tuning", n_trials=30):
    _stop_event.clear()

    if base_config is None:
        base_config = Config()

    def objective(trial):
        trial_params = {
            "interactions": trial.suggest_int("interactions", 0, 50),
            "outer_bags": trial.suggest_int("outer_bags", 10, 20),
            "inner_bags": trial.suggest_int("inner_bags", 0, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }

        config = Config(**{**vars(base_config), **trial_params})

        model = EBMModel(
            config=config, 
            task=task, 
            experiment_name=experiment_name, 
            run_name=f"trial_{trial.number}",
            feature_names=feature_names
        )

        return model.cross_validate(x_train, y_train, cv=3, scoring="neg_mean_squared_error").mean()
    
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
