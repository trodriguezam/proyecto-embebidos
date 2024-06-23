import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib

train_dir = 'dataset-v3/training'
validation_dir = 'dataset-v3/validation'

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 1)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid') # revisar params (# de clases, f de activacion)
])

optimizer = Adam(learning_rate=0.001) # revisar learning rate (+ = mas preciso, - = mas rapido)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # revisar loss function

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
class_mode='binary', # revisar class_mode
color_mode='grayscale',
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
validation_dir,
target_size=(96, 96),
batch_size=32,
class_mode='binary', # revisar class_mode
color_mode='grayscale',
)

batch_size = 32
total_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
total_validation_samples = sum([len(files) for _, _, files in os.walk(validation_dir)])
steps_per_epoch = (total_train_samples // batch_size)
validation_steps = (total_validation_samples // batch_size)

NUM_EPOCHS = 200 # revisar numero de epochs
history = model.fit(
train_generator,
steps_per_epoch=steps_per_epoch, # revisar steps_per_epoch (total de imagenes / batch_size)
epochs=NUM_EPOCHS,
verbose=1,
validation_data=validation_generator,
# validation_steps=validation_steps
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
output_path = 'Model accuracy.png'
plt.savefig(output_path)
plt.show()

# Save the model in the SavedModel format
export_dir = "saved_model/simpler_model"
model.export(export_dir)

# Convert to TFLite model with integer quantization
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        input_value, _ = next(validation_generator)
        yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()
tflite_model_file = pathlib.Path("simpler_model.tflite")
tflite_model_file.write_bytes(tflite_model)