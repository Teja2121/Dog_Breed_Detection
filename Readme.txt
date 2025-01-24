Dog Breed Detection Using Deep Learning

This project focuses on building a robust deep learning pipeline to classify images of dogs into 60 unique breeds. Leveraging a dataset of over 10,000 images, a transfer learning approach with the ResNet50V2 architecture pre-trained on ImageNet was implemented to achieve an accuracy of 81%.

Project Files
Dog_Breed_Detection_Python.py: Python script containing the entire pipeline, from data preprocessing to model training and prediction.
Dog_Breed_Detection.pdf: Detailed project report outlining the methodology, architecture, results, and insights gained.
Dataset: A collection of images organized into training and testing sets, accompanied by a CSV file (labels.csv) containing the corresponding breed labels.

Key Features
Dataset: Contains 10,000+ images across 60 dog breeds. Data is split into training (80%) and testing (20%) sets.
Model Architecture:
ResNet50V2 pre-trained on ImageNet with custom top layers including global average pooling, batch normalization, dropout, and a softmax activation layer.
Optimized using RMSprop optimizer with a learning rate of 1e-3.
Data Preprocessing:
Images resized to 224x224 pixels and normalized.
Labels encoded using LabelEncoder.
Data augmentation using ImageDataGenerator for transformations like rotation, zoom, and flips to improve generalization.
Performance: Achieved 81% accuracy on the test dataset.
Prediction Pipeline: Real-time breed prediction from uploaded dog images