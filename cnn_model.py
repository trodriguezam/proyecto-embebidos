import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import pathlib
import numpy as np

train_dir = 'dataset/training'
validation_dir = 'dataset/validation'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(96, 96, 1)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
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

batch_size = 32
total_train_samples = sum([len(files) for _, _, files in os.walk(train_dir)])
total_validation_samples = sum([len(files) for _, _, files in os.walk(validation_dir)])
steps_per_epoch = (total_train_samples // batch_size)
validation_steps = (total_validation_samples // batch_size)

NUM_EPOCHS = 1
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NUM_EPOCHS,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=validation_steps,
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


export_dir = "saved_models/model_tmp.keras"
model.save(export_dir)


converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        input_value, _ = next(validation_generator)
        yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()
tflite_model_file = pathlib.Path("saved_models/model_5/quant_model_5.tflite")
tflite_model_file.write_bytes(tflite_model)

tflite_model_path = 'saved_models/model_5/quant_model_5.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to evaluate the TFLite model
def evaluate_tflite_model(interpreter, validation_generator, steps):
    accuracy = 0
    total_samples = 0
    for step in range(steps):
        input_data, labels = next(validation_generator)
        # Process each image in the batch individually
        for i in range(input_data.shape[0]):  # Iterate through each image in the batch
            single_input_data = np.expand_dims(input_data[i], axis=0)  # Add batch dimension
            interpreter.set_tensor(input_details[0]['index'], single_input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data, axis=1)
            true_label = np.argmax(labels[i])
            accuracy += np.sum(prediction == true_label)
            total_samples += 1
    return accuracy / total_samples 

# Evaluate the model
validation_steps = len(validation_generator)
tflite_accuracy = evaluate_tflite_model(interpreter, validation_generator, validation_steps)
print(f'Post-quantization accuracy: {tflite_accuracy:.4f}')
