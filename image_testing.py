import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import numpy as np

validation = "dataset/validation"

class_names = ['cow', 'other', 'sheep']

def plot_image(i, predictions_array, true_labels, imgs):
    prediction_prob, true_label, img = predictions_array[i], true_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(prediction_prob)
    true_label = np.argmax(true_label)

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(prediction_prob),
                                         class_names[true_label]), color=color)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

validation_generator = validation_datagen.flow_from_directory(
    validation,
    target_size=(96, 96),
    class_mode='categorical',
    color_mode='grayscale'
)

tflite_model_file = pathlib.Path("saved_models/model_5/quant_model_5.tflite")

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

test_batches = validation_generator

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

predictions = []
test_labels, test_imgs = [], []

for _ in range(500):
    img, label = next(test_batches)
    single_image = img[0:1]
    interpreter.set_tensor(input_index, single_image)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))

    test_labels.append(label[0])
    test_imgs.append(single_image)

score = 0
for item in range(0, 500):
    prediction = np.argmax(predictions[item])
    true_label = np.argmax(test_labels[item])
    if prediction == true_label:
        score += 1

print("De 500 predicciones obtenemos " + str(score) + " correctas")

for index in range(0, 500):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index, predictions, test_labels, test_imgs)
    plt.show()