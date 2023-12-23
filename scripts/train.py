import os
import yaml
import click
import mlflow
import numpy as np
import mlflow.keras
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from mlflow.models.signature import ModelSignature, Schema

def create_model(number_of_classes):
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
        
    return model

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to the config file")

def task(config_file):
    print("===== Training Started =====")
    with open(f'{config_file}') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    EXPERIMENT_NAME = config['EXPERIMENT_NAME']
    DIR = config['DIR']
    EPOCHS = int(config['EPOCHS'])
    BATCH_SIZE = int(config['BATCH_SIZE'])
    URI = config['URI']
    TARGET_SIZE = config['TARGET_SIZE']
    SEED = int(config['SEED'])
    COLOR_MODE = config['COLOR_MODE']
    CLASS_MODE = config['CLASS_MODE'] 
    CLASSES = config['CLASSES']
    CLASSES = CLASSES.split(',')
    TRAINING_INSTANCES = int(config['TRAINING_INSTANCES'])
    VALIDATION_INSTANCES = int(config['VALIDATION_INSTANCES'])
    
    client = mlflow.tracking.MlflowClient();
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    number_of_classes = len(CLASSES)
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator()
    
    train_generator = train_gen.flow_from_directory(
        os.path.join(DIR, FOLDER),
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE
    )
    
    val_generator = val_gen.flow_from_directory(
        os.path.join(DIR, 'test'),
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        class_mode=CLASS_MODE
    )
    
    model = create_model(number_of_classes)
    
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="training") as mlrun:
        mlflow.set_tag("mlrun.Name", "training")
        
        os.makedirs(os.path.join(DIR, 'weights'), exist_ok=True)
        csv_logger = CSVLogger(os.path.join(DIR, 'weights', 'training.csv'), separator=',', append=False)
        checkpoint = ModelCheckpoint(os.path.join(DIR, 'weights', '{epoch}.h5'), monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
        
        history = model.fit_generator(
            train_generator, 
            steps_per_epoch= TRAINING_INSTANCES//BATCH_SIZE, 
            epochs=EPOCHS, 
            validation_data=val_generator, 
            validation_steps= VALIDATION_INSTANCES//BATCH_SIZE,
            callbacks=[csv_logger, checkpoint]
        )
        
        for epoch, val in history.items():
            for key, value in val.items():
                mlflow.log_metric(key, value, step=epoch)
        
        if 'URI' in config:
            del config['URI']
        
        mlflow.log_params(config)
        mlflow.log_artifacts(os.path.join(DIR, 'weights'), "weights")
        
        input_schema = Schema([{"name": "image", "type": "array", "item": {"type": "array", "item": "number"}}])
        output_schema = Schema([{"name": "emotion", "type": "string"}])
        model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        csv_path = os.path.join(DIR, 'weights', 'training.csv')
        df = pd.read_csv(csv_path)
        best_epoch = df['val_accuracy'].idxmax() + 1
        
        best_model_path = os.path.join(DIR, 'weights', f'{best_epoch}.h5')
        best_model = load_model(best_model_path)
        
        mlflow.keras.log_model(model, "model", signature=model_signature)
    print("===== Training Completed =====")
        
if __name__ == "__main__":
    task()


    
 
 
 