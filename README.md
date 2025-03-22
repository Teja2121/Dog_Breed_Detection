# Dog Breed Detection Using Deep Learning

This project builds a robust deep learning pipeline to classify images of dogs into 60 unique breeds. Using a dataset of over 10,000 images, the project leverages transfer learning with ResNet50V2 to achieve an accuracy of 81% on the test dataset.

---

## Table of Contents
- [Introduction](#introduction)
- [Project Files](#project-files)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Performance](#performance)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)

---

## Introduction

Accurate dog breed classification can have applications in veterinary medicine, pet adoption, and more. This project uses deep learning and transfer learning to create a reliable classifier capable of identifying 60 dog breeds from images.

---

## Project Files

- **Dog_Breed_Detection_Python.py**: Python script containing the entire pipeline, from data preprocessing to model training and prediction.
- **Dog_Breed_Detection.pdf**: Detailed report outlining the methodology, architecture, results, and insights.
- **Dataset**:
  - 10,000+ images organized into training and testing sets.
  - `labels.csv`: Contains the corresponding breed labels.

---

## Key Features

### Dataset
- 10,000+ images categorized into 60 unique dog breeds.
- Training set: 80% of the data.
- Testing set: 20% of the data.

### Model Architecture
- **Base Model**: ResNet50V2 pre-trained on ImageNet.
- **Custom Layers**:
  - Global Average Pooling.
  - Batch Normalization.
  - Dropout Layer (to prevent overfitting).
  - Fully Connected Layer with Softmax Activation for classification.
- **Optimizer**: RMSprop with a learning rate of 1e-3.

### Data Preprocessing
- Resized images to 224x224 pixels.
- Normalized pixel values for model compatibility.
- Encoded labels using `LabelEncoder`.
- Applied data augmentation with `ImageDataGenerator` for transformations:
  - Rotation
  - Zoom
  - Horizontal and vertical flips

### Performance
- **Accuracy**: Achieved 81% accuracy on the test dataset.
- **Prediction Pipeline**: Real-time breed prediction for uploaded dog images.

---

## Technologies Used

- **Python**: Programming language for implementation.
- **TensorFlow/Keras**: Deep learning framework for model development.
- **NumPy**: Data manipulation.
- **Pandas**: Dataset handling.
- **Matplotlib/Seaborn**: Data visualization.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Teja2121/Dog-Breed-Detection.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the `Dog_Breed_Detection_Python.py` script:
   ```bash
   python Dog_Breed_Detection_Python.py
   ```
4. Upload a dog image to see the breed prediction in real-time.

---

Feel free to contribute by raising issues or submitting pull requests to improve the project!
