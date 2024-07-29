import os

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from src import data_utils, config

data_utils.download_datasets(config.DATASET_ROOT_PATH)

# Dataset folder
DATASET_FOLDER = os.path.join(config.DATASET_ROOT_PATH, config.DATASET_FILENAME)
DATASET_FOLDER = os.path.join(config.DATASET_ROOT_PATH, "eu-car-dataset_subset")

img_height = 224
img_width = 224
batch_size = 32

# Load train and test datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(DATASET_FOLDER, "train"),
    labels="inferred",
    label_mode="categorical",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    directory=os.path.join(DATASET_FOLDER, "test"),
    labels="inferred",
    label_mode="categorical",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names

# Configure data loader for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define the input shape
img_height, img_width = 224, 224  # Asegúrate de definir estos valores
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

# Data augmentation
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.6, 1.4)),
        layers.Lambda(lambda x: tf.image.random_hue(x, 0.2)),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomCrop(height=img_height, width=img_width),
        layers.RandomHeight(factor=0.1),
        layers.RandomWidth(factor=0.1)
    ]
)

# Apply data augmentation
augmented_inputs = data_augmentation(inputs)

# Preprocess inputs for InceptionV3
preprocessed_inputs = inception_preprocess_input(augmented_inputs)

# Load the InceptionV3 model with pre-trained ImageNet weights, excluding the top layers
inception = InceptionV3(
    include_top=False, 
    input_shape=preprocessed_inputs.shape[1:], 
    weights='imagenet')

# Descongelar las últimas capas del modelo preentrenado
for layer in inception.layers[-10:]:
    layer.trainable = True

# Create the model
inception_model = tf.keras.Sequential()
inception_model.add(inception)
inception_model.add(layers.GlobalAveragePooling2D())  # Cambiar Flatten por GlobalAveragePooling2D
inception_model.add(layers.Dropout(0.5))
inception_model.add(layers.Dense(len(class_names), activation='softmax'))

# Print a summary of the model architecture
print(inception_model.summary())

# Compile the model
inception_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Ajusta la tasa de aprendizaje
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Define callbacks
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
inception_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,  # Incrementa el número de épocas
    callbacks=[lr_scheduler, early_stopping]
)
