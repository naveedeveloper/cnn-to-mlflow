import os
import yaml
import click
import mlflow
import numpy as np
import mlflow.keras
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to the config file")
@click.option("--run_id", default=None, help="Run ID of the best model")

def task(config_file, run_id):
    print("===== Registering Model =====")
    with open(f'{config_file}') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    EXPERIMENT_NAME = config['EXPERIMENT_NAME']
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    model = client.get_registered_model(name=run_id)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="registering") as mlrun:
        mlflow.set_tag("mlrun.Name", "registering")

        if model:
            # If model already exists, increment version number
            version = int(model.latest_versions[0].version)
            version += 1
            version = str(version)
        else:
            # If the model doesn't exist, register it
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=run_id,
                tags={"version": "1", "staging": "Archived"},
                description="Your model description",
                archive=True
            )
            version = "1"  # Initial version for a new model

        mlflow.register_model(
            model_uri=model.source,
            name=run_id,
            tags={"version": version, "staging": "Staging"},
            description=model.description
        )

    print("===== Model Registered =====")

if __name__ == "__main__":
    task()