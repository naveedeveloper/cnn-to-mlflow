import yaml
import click
import mlflow

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to the config file")

def run_pipeline(config_file):
    print("===== Starting Pipeline =====")
    with open(f'{config_file}') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    EXPERIMENT = config['EXPERIMENT_NAME']
    STEPS = config['PIPELINE_STEPS']
    URI = config['URI']
    
    active_steps = STEPS.split(",")
    mlflow.set_tracking_uri(URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    
    if experiment == None:
        experiment = mlflow.create_experiment(EXPERIMENT)
        
        if experiment != None:
            print("===== Experiment created sucessfully =====")
    
    run = client.create_run(experiment_id=experiment.experiment_id, run_name="Pipeline")
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True) as mlrun:
        if "augment" in active_steps:
            mlflow.run(".", "augment", params={})
        
        if "train" in active_steps:
            run = mlflow.run(".", "train", params={})
            run_id = run.run_id
            
        if "register" in active_steps:
            mlflow.run(".", "register", params={"run_id": run_id})
    print("===== Pipeline Finished =====")

if __name__ == "__main__":
    run_pipeline()