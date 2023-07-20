# Face Detection Deep Learning Model
This project focuses on building a face detection deep learning model using a custom dataset. The dataset comprises 90 images of the project creator's face, each augmented 60 times to increase diversity and improve model generalization. The model uses two distinct losses - binary cross-entropy loss for face detection and a custom localization loss for bounding box prediction.

# Data Preprocessing
Data Collection: A collection of 90 facial images of the project creator were obtained. These images served as the foundation for training the face detection model.

Data Augmentation: Each of the 90 images was augmented 60 times using techniques like rotation, scaling, flipping, and brightness adjustments. Augmentation enhances the diversity of the dataset and helps the model learn robust features.

Train-Test-Validation Split: The augmented dataset was divided into three subsets: training set (80%), test set (10%), and validation set (10%). The training set is used to train the model, the test set is employed for unbiased evaluation, and the validation set aids in hyperparameter tuning.

# Deep Neural Network Architecture
Convolutional Neural Network (CNN): The face detection model is based on a CNN architecture. CNNs are ideal for image-related tasks as they can effectively capture hierarchical features from the input images.

Face Detection Head: The first part of the model focuses on detecting whether a face is present in the image or not. This binary classification task utilizes binary cross-entropy loss, which measures the difference between the predicted probability and the ground truth for face detection.

Bounding Box Prediction Head: The second part of the model predicts the bounding box coordinates around the detected face (if any). This localization task employs a custom localization loss, which penalizes the model based on the discrepancies between predicted and ground truth bounding box coordinates.

# Training and Losses
Model Training: The model was trained on the training set using the Adam optimizer, adjusting the weights to minimize the combined loss of face detection and bounding box prediction.

Binary Cross-Entropy Loss (BCE Loss): BCE loss is used to train the face detection head. It calculates the difference between the predicted probability (sigmoid output) and the ground truth label (0 or 1) for each image. This loss encourages the model to accurately classify images into either face or non-face categories.

Custom Localization Loss: The bounding box prediction head is trained using a custom localization loss. This loss function measures the difference between predicted bounding box coordinates (x, y, width, height) and the ground truth bounding box coordinates. It encourages the model to accurately localize the face within the image.

# Evaluation
Model Evaluation Metrics: The trained model is evaluated on the test set using metrics such as accuracy for face detection and Intersection over Union (IoU) for bounding box prediction.

Accuracy: Accuracy measures the model's ability to correctly identify faces in the images and is calculated as the ratio of correctly predicted faces to the total number of faces.

Intersection over Union (IoU): IoU is used to evaluate the accuracy of the bounding box predictions. It measures the overlap between the predicted bounding box and the ground truth bounding box. A high IoU indicates a better localization performance.

# Conclusion
By developing a face detection deep learning model with custom data augmentation and using two different losses, this project achieves accurate face detection and bounding box prediction. The evaluation metrics demonstrate the model's efficacy in detecting faces and localizing them accurately within the images. This model can be further deployed for various face detection applications, contributing to the development of advanced computer vision solutions
