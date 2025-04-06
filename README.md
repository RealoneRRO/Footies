# Footwear Classifier
This resource gives a starting example for running TensorFlow training tasks on GPU inside Saturn Cloud. I trained a deep learning model with TensorFlow on a single GPU. The model was built for a student entrepreneur's small-scale business on campus to help classify different types of footwear. Trying to solve the business problem, I picked up a dataset on Kaggle. The dataset comprises three classes of footwear rampant among the students on campus, cleaned the data, and divided images into three subfolders, namely: Train, Validation, and Test. The exploration data analysis had done well to be able to train the Deep Learning (DL) classification model. 

## The Aim and Objectives
This model aims to classify different types of footwear worn by the students on campus for Soft Footies. The following are the objectives of the projects:

### Data Collection & Preprocessing
* Downloading dataset of footwear images.
* Resizing and normalizing of images for consistent model input.
* Setting augmentation (Zoom range, horizontal flipping, and shearing) for performance improvement and robustness.
### Model Training & Evaluation
* Use python libraries such as TensorFlow, Keras, NumPy, and MatplotLib.
* Use a Convolutional Neural Network (CNN) model, Xception, trained on ImageNet.
* Evaluate accuracy, precision, recall, and F1-score.
* Adding more layers to the pre-trained models: **Rectified Linear Unit** (relu) activation function.
  
### Hyperparameter Fine-tuning & Visualization
* Experiment with hyperparameters (batch size, seed, learning rate).
* Regularization and Dropout to prevent overfitting.
* Training Larger models and using a significantly high number of epochs

## Extra Resources
* [Multi-GPU TensorFlow on Saturn Cloud](https://saturncloud.io/blog/tensorflow_intro/)
* [Overview on GPUs](https://saturncloud.io/docs/reference/intro_to_gpu/)
* [Learn more about TensorFlow](https://www.tensorflow.org/)
* Prefer to use PyTorch? [Visit our docs about that!](https://saturncloud.io/docs/examples/pytorch/)
