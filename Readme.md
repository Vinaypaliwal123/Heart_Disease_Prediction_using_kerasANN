# 🫀 Heart Disease Prediction Model 🫀
## Overview

Welcome to the **Heart Disease Prediction Model** repository! In this project, we've built an intelligent model that analyzes various health-related features to predict the likelihood of heart disease. Whether you're a data enthusiast, a curious coder, or a medical aficionado, this project has something for you.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Making Predictions](#making-predictions)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Cardiovascular diseases are a leading cause of mortality worldwide. Early detection and accurate prediction play a crucial role in preventing heart-related issues. Our model leverages machine learning techniques to provide insights into whether an individual is at risk of heart disease.

## Installation

To get started, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vinaypaliwal123/Heart_Disease_Prediction_using_kerasANN
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn tensorflow
   ```

## Dataset

Our dataset (`dataset1.csv`) contains a wealth of information about patients. Here are some key features:

- **Age**: The age of the patient.
- **Sex**: Gender (0 for female, 1 for male).
- **Blood Pressure**: Systolic blood pressure.
- **Cholesterol Levels**: Serum cholesterol levels.
- **ECG Results**: Electrocardiogram results (0, 1, or 2).
- **Maximum Heart Rate**: Maximum heart rate achieved during exercise.
- **Exercise-Induced Angina**: Presence of angina induced by exercise.
- **ST Depression**: ST depression induced by exercise relative to rest.
- **Number of Major Vessels**: Colored by fluoroscopy.
- **Target**: Binary label (0 for no heart disease, 1 for heart disease).

## Model Architecture

Our neural network architecture is designed for heart disease classification:

1. **Input Layer**: Flattened input features.
2. **Hidden Layers**:
   - Layer 1: 20 neurons with ReLU activation.
   - Layer 2: 40 neurons with ReLU activation.
   - Layer 3: 20 neurons with ReLU activation.
3. **Output Layer**: 2 neurons with softmax activation (binary classification).

## Training and Evaluation

We trained the model using the Adam optimizer and sparse categorical cross-entropy loss. The training history is stored in `history`.

- **Accuracy on Test Data**: {{accuracy}} (Impressive, right?)

## Making Predictions

Want to predict heart disease for a new patient? Follow these steps:

1. Create an input data point (e.g., `(34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2)`).
2. Reshape and standardize the input data.
3. Use our trained model to predict the likelihood of heart disease.


### 1. Data Loading and Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset from 'dataset1.csv'
dataset = pd.read_csv('dataset1.csv')

# Display the first few rows of the dataset
print(dataset.head())

# Check the shape of the dataset (number of rows and columns)
print(dataset.shape)

# Get information about the dataset (data types, non-null counts, etc.)
print(dataset.info())

# Check for missing values
print(dataset.isnull().sum())

# Summary statistics
print(dataset.describe())

# Count the occurrences of each target class
print(dataset['target'].value_counts())
```

Explanation:
- You've imported necessary libraries (`pandas`, `numpy`, `matplotlib`, and `train_test_split`).
- The dataset is loaded from the CSV file ('dataset1.csv') using `pd.read_csv`.
- You've inspected the first few rows of the dataset using `dataset.head()`.
- The shape of the dataset (rows and columns) is checked using `dataset.shape`.
- Information about the dataset (data types, non-null counts) is obtained using `dataset.info()`.
- Missing values are identified using `dataset.isnull().sum()`.
- Summary statistics (mean, min, max, etc.) are computed using `dataset.describe()`.
- The distribution of the target variable ('target') is displayed using `dataset['target'].value_counts()`.

### 2. Data Preprocessing

```python
# Separate features (X) and target variable (Y)
X = dataset.drop(columns='target', axis=1)
Y = dataset['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Standardize features using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.values)
X_test = sc.transform(X_test.values)
```

Explanation:
- You've separated features (X) and the target variable (Y).
- The data is split into training and testing sets using `train_test_split`.
- Features are standardized using `StandardScaler`.

### 3. Neural Network Model

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(13,)), 
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10)
```

Explanation:
- You've created a neural network model using Keras.
- The architecture includes:
    - Input layer (Flatten) with 13 features.
    - Three hidden layers with ReLU activation functions.
    - Output layer with softmax activation for binary classification.
- The model is compiled with 'adam' optimizer and 'sparse_categorical_crossentropy' loss.
- Training is performed using `model.fit`.

### 4. Model Evaluation

```python
# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

Explanation:
- The model's performance is evaluated on the test data using `model.evaluate`.
- The test accuracy is printed.

### 5. Making Predictions

```python
# Make predictions on test data
Y_pred = model.predict(X_test)
Y_pred_labels = [np.argmax(i) for i in Y_pred]

# Example input data for prediction
input_data = np.array([34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2])
input_data = input_data.reshape(1, -1)
input_data = sc.transform(input_data)

# Predict for the input data
prediction = model.predict(input_data)
prediction_label = np.argmax(prediction)

if prediction_label == 0:
    print('NO Heart Disease')
else:
    print('Heart Disease')
```

Explanation:
- You've made predictions on the test data and converted predicted probabilities to class labels.
- An example input data point is created and standardized.
- The model predicts whether it corresponds to heart disease or not.

## Contributing

Contributions are heartily welcome! Whether it's code improvements, bug fixes, or feature enhancements, feel free to contribute. Let's save lives together! 🙌
