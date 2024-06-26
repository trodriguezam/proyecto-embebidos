import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib

train_dir = 'dataset/training'
validation_dir = 'dataset/validation'

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.50, final_sparsity=0.90, begin_step=2000, end_step=4000)
}

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(96, 96, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=2000, end_step=4000)
}

pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)

pruned_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

optimizer = Adam(learning_rate=0.0001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
)

history = pruned_model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model = sparsity.strip_pruning(pruned_model)
model.save('saved_models/optimized_model/optimized_model.keras')

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_file = pathlib.Path("saved_models/optimized_model/optimized_model.tflite")
tflite_model_file.write_bytes(tflite_model)

# Now use the TFLite model on the ESP32-CAM
