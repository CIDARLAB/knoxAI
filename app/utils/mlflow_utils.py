from mlflow.tracking import MlflowClient

def create_train_run(experiment_name: str, run_name: str) -> str:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id=experiment_id, tags={"mlflow.runName": run_name})
    return run.info.run_id
