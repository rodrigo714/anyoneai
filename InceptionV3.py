import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from src import data_utils, config
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

data_utils.download_datasets(config.DATASET_ROOT_PATH)

# Dataset folder
DATASET_FOLDER = os.path.join(config.DATASET_ROOT_PATH, config.DATASET_FILENAME)
DATASET_FOLDER = os.path.join(config.DATASET_ROOT_PATH, "eu-car-dataset_subset")

img_height = 224
img_width = 224
batch_size = 256

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
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),                           # Volteo horizontal y vertical
        layers.RandomRotation(0.5),                                             # Rotación aleatoria
        layers.RandomZoom(0.5),                                                 # Zoom aleatorio
        layers.RandomBrightness(0.5),                                           # Brillo aleatorio
        layers.RandomContrast(0.5),                                             # Contraste aleatorio
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.6, 1.4)),       # Saturación aleatoria
        layers.Lambda(lambda x: tf.image.random_hue(x, 0.2)),                   # Tono aleatorio
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),          # Traslación aleatoria
        layers.RandomCrop(height=img_height, width=img_width),                  # Recorte aleatorio
        layers.RandomHeight(factor=0.5),                                        # Altura aleatoria
        layers.RandomWidth(factor=0.5),                                         # Ancho aleatorio
        layers.GaussianNoise(0.5),                                              # Ruido gaussiano
        layers.GaussianDropout(0.5),
        layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.1)),  # Brillo aleatorio (versión alternativa)
        layers.Lambda(lambda x: tf.image.random_flip_left_right(x)),            # Volteo aleatorio izquierdo-derecho
        layers.Lambda(lambda x: tf.image.random_flip_up_down(x))                # Volteo aleatorio arriba-abajo
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

# Set all layers to non-trainable
for layer in inception.layers:
    layer.trainable = False

# Set the last 10 layers to trainable
for layer in inception.layers:
    layer.trainable = False

# Create the model
inception_model = tf.keras.Sequential()
inception_model.add(inception)
inception_model.add(layers.GlobalAveragePooling2D())
inception_model.add(layers.BatchNormalization())
# inception_model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# inception_model.add(layers.Dropout(0.6))
# inception_model.add(layers.BatchNormalization())
# inception_model.add(layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# inception_model.add(layers.Dropout(0.6))
# inception_model.add(layers.BatchNormalization())
# inception_model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
# inception_model.add(layers.Dropout(0.6))
# inception_model.add(layers.BatchNormalization())
inception_model.add(layers.Dense(len(class_names), activation='softmax'))

# Print a summary of the model architecture
print(inception_model.summary())

# Compile the model
inception_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Define callbacks
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = inception_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=50,
    callbacks=[lr_scheduler, early_stopping]
)

# Access training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Reporting model results
y_true = np.concatenate([y for x, y in test_ds], axis=0)
if y_true.ndim > 1 and y_true.shape[1] > 1:
    y_true = np.argmax(y_true, axis=1)

y_pred = inception_model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap='Blues', ax=ax)

plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.show()