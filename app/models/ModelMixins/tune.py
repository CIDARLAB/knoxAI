import optuna
import mlflow
import threading

_stop_event = threading.Event()

def stop_tuning():
    _stop_event.set()

def tune_pytorch(train_json, val_json, task, model_class, model_config, config=None, experiment_name="tuning", n_trials=30):
    _stop_event.clear()

    trial_base = model_config(**{**vars(config), "max_epochs": 150})

    def objective(trial):
        trial_params = get_trial_params(trial, model_class)

        config = model_config(**{**vars(trial_base), **trial_params})

        model = model_class(
            config=config, 
            task=task,
            experiment_name=experiment_name, 
            run_name=f"optuna_trial_{trial.number}" # TODO: Add Prefix
        )

        model.train(
            train_json=train_json,
            val_json=val_json,
            save_model=False,
            patience=10,
            min_delta=1e-4
        )
        
        # Return val metric to minimise — pull from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model.run_id)
        trial.set_user_attr("run_id", model.run_id)
        return run.data.metrics["best_val_loss"]  # or val_loss
    
    def stop_callback(study, trial):
        if _stop_event.is_set():
            study.stop()

    def cleanup_runs(study):
        client = mlflow.tracking.MlflowClient()
        for trial in study.trials:
            if trial.number != study.best_trial.number:
                client.delete_run(trial.user_attrs["run_id"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_callback])
    #cleanup_runs(study)

    if not study.trials:
        return {"status": "stopped before any trials completed"}

    best = study.best_trial
    return {
        "best_params": best.params,
        "best_value": best.value,   # best validation loss
        "n_trials": len(study.trials),
    }

def tune(x_train, y_train, task, model_class, model_config, config=None, feature_names=None, experiment_name="tuning", cv=5, n_trials=30):
    _stop_event.clear()

    trial_base = model_config(**{**vars(config)})

    def objective(trial):
        trial_params = get_trial_params(trial, model_class)

        config = model_config(**{**vars(trial_base), **trial_params})

        model = model_class(
            config=config, 
            task=task, 
            experiment_name=experiment_name, 
            run_name=f"optuna_trial_{trial.number}", # TODO: Add Prefix
            feature_names=feature_names
        )

        trial.set_user_attr("run_id", model.run_id)
        return model.cross_validate(x_train, y_train, cv=cv, scoring="neg_mean_squared_error").mean()
    
    def stop_callback(study, trial):
        if _stop_event.is_set():
            study.stop()

    # Delete run from MLflow if it is not the best trial, to avoid cluttering the MLflow experiment with unnecessary runs.
    def cleanup_runs(study):
        client = mlflow.tracking.MlflowClient()
        for trial in study.trials:
            if trial.number != study.best_trial.number:
                client.delete_run(trial.user_attrs["run_id"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_callback])
    #cleanup_runs(study)

    if not study.trials:
        return {"status": "stopped before any trials completed"}

    best = study.best_trial
    return {
        "best_params": best.params,
        "best_value": best.value,
        "n_trials": len(study.trials)
    }

def get_trial_params(trial, model_class):
    if model_class.__name__ == "TransformerModel":
        return {
            "model_dim"   : trial.suggest_categorical("model_dim", [64, 128, 256]),
            "num_heads"   : trial.suggest_categorical("num_heads", [4, 8]),
            "num_layers"  : trial.suggest_int("num_layers", 2, 4),
            "dropout"     : trial.suggest_float("dropout", 0.0, 0.3),
            "lr"          : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size"  : trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    elif model_class.__name__ == "GNNModel":
        return {
            "hidden_dim"   : trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            "num_layers"   : trial.suggest_int("num_layers", 2, 5),
            "dropout"      : trial.suggest_float("dropout", 0.0, 0.3),
            "lr"           : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "aggr"         : trial.suggest_categorical("aggr", ["sum", "mean", "max"]),
            "graph_conv"   : trial.suggest_categorical("graph_conv", ["graph", "nn", "gat", "tconv"]),
            "batch_size"   : trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    elif model_class.__name__ == "MLPModel":
        return {
            "hidden_dim"   : trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            "num_layers"   : trial.suggest_int("num_layers", 4, 8),
            "dropout"      : trial.suggest_float("dropout", 0.0, 0.3),
            "lr"           : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size"   : trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    elif model_class.__name__ == "RandomForestModel":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_categorical("max_depth", [10, 20, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", [0.15, "sqrt", "log2"]),
        }
    elif model_class.__name__ == "XGBoostModel":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.3, 0.9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
    elif model_class.__name__ == "EBMModel":
        return {
            "interactions": trial.suggest_int("interactions", 0, 50),
            "outer_bags": trial.suggest_int("outer_bags", 10, 20),
            "inner_bags": trial.suggest_int("inner_bags", 0, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    else:
        raise ValueError(f"Unsupported model class: {model_class.__name__}")
    
