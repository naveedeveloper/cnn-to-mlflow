import yaml
import click
import mlflow
import random
from PIL import Image, ImageEnhance

def random_rotation(image_path, max_angle=30):
    image = Image.open(image_path)
    angle = random.uniform(-max_angle, max_angle)
    rotated_image = image.rotate(angle)
    return rotated_image

def random_crop(image_path, crop_size=(200, 200)):
    image = Image.open(image_path)
    width, height = image.size
    left = random.randint(0, width - crop_size[0])
    top = random.randint(0, height - crop_size[1])
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def horizontal_flip(image_path):
    image = Image.open(image_path)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_image

@click.command()
@click.option("--config_file", default="config.yaml", help="Path to the config file")

def task(config_file):
    print("===== Starting Augmentation =====")
    with open(f'{config_file}') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    DIR = config['DIR']
    FOLDER = config['FOLDER']
    AUG_STEPS = config['AUG_STEPS']
    
    aug_steps = AUG_STEPS.split(',')
    os.makedirs(os.path.join(DIR, 'augmented'), exist_ok=True)
    folders = glob.glob(os.path.join(DIR, FOLDER, '*'))

    augmentation_functions = {
        'rotated': random_rotation,
        'cropped': random_crop,
        'flipped': horizontal_flip
    }

    for step in aug_steps:
        print(f"===== Starting {step} augmentation =====")
        os.makedirs(os.path.join(DIR, step), exist_ok=True)

        for f in folders:
            os.makedirs(os.path.join(DIR, step, os.path.basename(f)), exist_ok=True)

            for img in tqdm(glob.glob(os.path.join(f, '*'))):
                augmented_image = augmentation_functions[step](img)
                augmented_image.save(os.path.join(DIR, step, os.path.basename(f), os.path.basename(img)))

        FOLDER = step
        print(f"===== Finished {step} augmentation =====")
    
    with mlflow.start_run(run_name="augmentation") as mlrun:
        mlflow.set_tag("mlrun.Name", "augmentation")
        mlflow.log_artifacts(os.path.join(DIR, 'flipped'), artifact_path="augmented")
        
    print("===== Augmentation Finished =====")

if __name__ == "__main__":
    task()