# CNNtoMLFlow: Streamlined CNN Migration
- CNNtoMLFlow is a project designed to facilitate the seamless migration of Convolutional Neural Network (CNN) models to MLflow.
- This project is divided into three key steps:
  - Augmentation
  - Model Training
  - Model Registration.

## Project Steps
### 1. Augmentation
- The first step in the CNNtoMLFlow project involves data augmentation to enhance the training dataset.
- This augmentation includes rotation, cropping, and flipping of the input images.
- Augmentation is a crucial preprocessing step that helps diversify the dataset, leading to a more robust and generalized CNN model.

### 2. Training
- In the training step, the augmented dataset is used to train the CNN model.
- During this process, key parameters, metrics, and the trained model itself are logged to MLflow.
- This ensures comprehensive tracking of the model's performance and training details.
- Logging to MLflow provides a centralized and organized way to manage experiments and compare different model iterations.

### 3. Model Registration
- The final step involves registering the trained CNN model in MLflow.
- Model registration is essential for versioning and maintaining a record of different model iterations.
- This step simplifies the deployment process and allows for easy retrieval of specific models for inference or further experimentation.
