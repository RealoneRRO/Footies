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

## Techniques
Training deep learning models, especially for image classification tasks, is greatly enhanced by using transfer learning. Pretrained models like Xception, ResNet, or MobileNet, already trained on large datasets like ImageNet, can be fine-tuned on the specific dataset to save time and improve performance. Initially, I did a lot of image resizing and normalizing to see which size had the best accuracy and more consistent between 150x150 and 299x299 image size. ImageNet is used to pre-train the Xception model with an input size of (299, 299, 3) that enhances effective weight leveraging for ImageNet and preserves more intricate features' feasibility. You can downscale the supplied image by passing it an image size of (150, 150, 3). The input will be accepted by the model, however there may be problems:

-Information Loss: Because the Xception model is designed for larger images, details may be lost in smaller images.  Particularly for complex tasks, it might not capture fine-grained features, which could lower model accuracy.
-Less Spatial Detail: With smaller images, Xception’s deep convolutional layers won't be able to extract features with as much precision because there’s less spatial data to process.
Overall, **Global Average Pooling** (GAP) computes the average of each feature map, and the result will have a size proportional to the input image dimensions; that is the smaller input size will produce a smaller output after each convolution and pooling operation, and it might be enough for simpler models but obviously not for complex and larger model like this. Alongside this, applied data augmentation (like rotation, flipping, zooming, and brightness changes) during training increases data variability, helping the model generalize better and avoid overfitting.

Balancing overfitting and underfitting fewer epochs is usually not enough to significantly increase epochs from 30 to 50 to better fine-tune the hyperparameter(seed=1, learning_rate=0.01, droprate=0.05 layer=100) and added more layers to fit the model. Initially, you can freeze the base layers to train only the top classifier, then gradually unfreeze and fine-tune more layers. Training more Epochs and layers can increase time and accuracy with the help of an activation function called relu. It avoids saturation and helps neural networks learn complex patterns and speed up the training process. It involves smart techniques like hyperparameter visualization to pick the best parameter for the model, where the learning rate is gradually reduced as training progresses using tools like Matplotlib. Early stopping is another critical method, which halts training when validation loss no longer improves, preventing overfitting and saving time. Additionally, regularization techniques such as Dropout and L2 regularization are useful in reducing model complexity and improving generalization. Batch normalization, batch_size=32, placed between layers, helps stabilize learning and allows for higher learning rates.

Choosing the right optimizer is also essential as it is widely used for its ability to adjust learning rates adaptively. Visualizing training and validation accuracy/loss with plots helps identify issues like underfitting or overfitting. Always ensure a proper split between training and validation sets for honest evaluation. For performance boosts on compatible hardware, each hyperparameter selected was based on the highest training and validation accuracy. Mixed precision training can be enabled to accelerate computations while maintaining accuracy. Together, these techniques create a strong and reliable foundation for training deep learning models effectively. Using the model in the end, the image was loaded to test the model, generating an array in the form of a tuple (batch_size=1, height=299, width=299, no_of_class=3), and the model predicted the right class. The image of a shoe is loaded to the model.


## In Conclusion
The deep learning model built with the Xception architecture effectively utilizes transfer learning by repurposing a pretrained network to solve a custom image classification task. By freezing the base model and adding a lightweight classification head, the model benefits from the learned features of a large-scale dataset (ImageNet) while focusing on learning the unique features of our specific dataset. Techniques like dropout, data augmentation, and early stopping were key in enhancing the model's generalization and preventing overfitting during training. In conclusion, the model architecture, preprocessing pipeline, and training strategies come together to form a high-performing, efficient, and scalable deep learning solution for image classification. It’s flexible enough for fine-tuning and can be adapted to similar real-world computer vision problems with ease. Whether for production deployment or continued experimentation, this model provides a solid foundation for accurate and intelligent visual recognition systems.


## Extra Resources
* [Multi-GPU TensorFlow on Saturn Cloud](https://saturncloud.io/blog/tensorflow_intro/)
* [Overview on GPUs](https://saturncloud.io/docs/reference/intro_to_gpu/)
* [Learn more about TensorFlow](https://www.tensorflow.org/)
* Prefer to use PyTorch? [Visit our docs about that!](https://saturncloud.io/docs/examples/pytorch/)
