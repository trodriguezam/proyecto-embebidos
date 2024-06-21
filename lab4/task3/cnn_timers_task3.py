import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import numpy as np

# Paths to the training and validation datasets
train_dir = 'dataset-v2/training'
validation_dir = 'dataset-v2/validation'

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale',
)

# Calculate steps per epoch
batch_size = 32
total_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
total_validation_samples = sum([len(files) for _, _, files in os.walk(validation_dir)])
steps_per_epoch = (total_train_samples // batch_size)
validation_steps = (total_validation_samples // batch_size)

# Measure time for data preprocessing
start_preprocessing = time.time()

# Preprocessing data
for _ in range(steps_per_epoch):
    next(train_generator)

end_preprocessing = time.time()
preprocessing_time = end_preprocessing - start_preprocessing

# Train the model and measure time
NUM_EPOCHS = 200
start_training = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

end_training = time.time()
training_time = end_training - start_training

# Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Measure time for model evaluation
start_evaluation = time.time()

# Evaluate the model
model.evaluate(validation_generator)

end_evaluation = time.time()
evaluation_time = end_evaluation - start_evaluation

# Print times
print(f"Preprocessing time: {preprocessing_time} seconds")
print(f"Training time: {training_time} seconds")
print(f"Evaluation time: {evaluation_time} seconds")

# Calculate total time
total_time = preprocessing_time + training_time + evaluation_time

# Assume that the training time is the parallelizable part
parallelizable_time = training_time

# Calculate the parallelizable fraction P
P = parallelizable_time / total_time

print(f"Fracción paralelizable P: {P}")

# Define the range of cores
cores = np.arange(1, 65)  # From 1 to 64 cores

# Calculate the acceleration for each number of cores using Amdahl's Law
acceleration = 1 / ((1 - P) + (P / cores))

# Plot the number of cores vs acceleration
plt.figure(figsize=(10, 6))
plt.plot(cores, acceleration, marker='o')
plt.xlabel('Number of Cores')
plt.ylabel('Acceleration')
plt.title('Number of Cores vs Acceleration')
plt.grid(True)
output_path = 'task3.png'
plt.savefig(output_path)
plt.show()


