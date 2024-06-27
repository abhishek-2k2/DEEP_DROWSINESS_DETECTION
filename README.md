<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
 
</head>
<body>

<h1>Face Mask Detection</h1>
<p>This project aims to detect whether a person is wearing a face mask using machine learning techniques.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
</ul>

<h2 id="introduction">Introduction</h2>
<p>Face mask detection is an important task for public safety, especially during the COVID-19 pandemic. This project uses machine learning to classify images into two categories: with mask and without mask.</p>

<h2 id="dataset">Dataset</h2>
<p>The dataset used in this project is the Face Mask Dataset from Kaggle, which contains images of people with and without face masks.</p>
<p>Dataset link: <a href="https://www.kaggle.com/datasets/omkargurav/face-mask-dataset" download>Face Mask Dataset</a></p>

<h2 id="installation">Installation</h2>
<p>To run this project locally, follow these steps:</p>
<pre><code>
!pip install kaggle
# configure Kaggle API credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# download dataset from Kaggle
!kaggle datasets download -d omkargurav/face-mask-dataset
# unzip the dataset
from zipfile import ZipFile
dataset = '/content/face-mask-dataset.zip'
with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
</code></pre>

<h2 id="usage">Usage</h2>
<p>Load and preprocess the dataset:</p>
<pre><code>
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Load images with masks
with_mask_files = os.listdir('/content/data/with_mask')
data = []
for img_file in with_mask_files:
    image = Image.open('/content/data/with_mask/' + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Load images without masks
without_mask_files = os.listdir('/content/data/without_mask')
for img_file in without_mask_files:
    image = Image.open('/content/data/without_mask/' + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Create labels
with_mask_labels = [1] * len(with_mask_files)
without_mask_labels = [0] * len(without_mask_files)
labels = with_mask_labels + without_mask_labels

# Convert to numpy arrays
X = np.array(data)
Y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0
</code></pre>

<h2 id="results">Results</h2>
<p>Train a neural network model to classify images:</p>
<pre><code>
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, Y_train, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, Y_test)
print('Test accuracy:', test_acc)
</code></pre>

</body>
</html>
