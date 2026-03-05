# Multiclass-Fish-Image-Classification
# 

# Introduction:
This project focuses on classifying fish images into multiple categories using deep learning techniques. A Convolutional Neural Network (CNN) is trained from scratch and compared with five pre-trained models using transfer learning. The best-performing model is saved and deployed using a Streamlit web application for real-time predictions.

# Domain:
Computer Vision | Deep Learning | Image Classification

# Business Use Cases:
- Enhanced fish species identification accuracy
- Deployment-ready AI system
- Model comparison for optimal architecture selection

# Skills Takeaway
Deep Learning
CNN Architecture
Transfer Learning
Image Augmentation
Model Evaluation Metrics
Confusion Matrix
Model Deployment
Streamlit
TensorFlow / Keras

# TECHNOLOGY USED
Python
TensorFlow
Streamlit

# Packages and Libraries
tensorflow
numpy
matplotlib
seaborn
streamlit
sklearn
PIL

# Understand the Dataset:
The dataset consists of fish images organized into folders by species. TensorFlow’s ImageDataGenerator is used to efficiently load and preprocess images.

# Preprocessing:
Images are rescaled to [0,1] range. Data augmentation techniques such as rotation, zoom, and horizontal flipping are applied to improve generalization.

# Model Building:
- CNN trained from scratch
- VGG16
- ResNet50
- MobileNet
- InceptionV3
- EfficientNetB0

Transfer learning is used by freezing base layers and adding custom classification layers.

# Model Evaluation:
Models are evaluated using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

The best-performing model is saved as best_model.h5.

# Develop Streamlit:
A Streamlit web application allows users to upload fish images and receive:
Predicted fish species
Confidence score
Interactive UI for real-time classification

# Analysis:
Transfer learning models outperform CNN trained from scratch.
EfficientNetB0 / ResNet50 typically provide highest accuracy.
The project demonstrates end-to-end deployment of a deep learning classification system.
