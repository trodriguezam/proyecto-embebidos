import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import numpy as np

validation = "project_imgs"

class_names = ['cow', 'sheep']

def plot_image(i, predictions_array, true_labels, imgs):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label)  # Asegúrate de obtener el índice de la etiqueta verdadera

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]), color=color)
    
def display_image_from_bytes(image_bytes, shape):
    # Convertir los bytes a una matriz numpy
    image = np.frombuffer(image_bytes, dtype=np.float32).reshape(shape)

    # Visualizar la imagen
    plt.imshow(image, cmap='gray')
    plt.show()


validation_datagen = ImageDataGenerator(
    rescale=1./255,
)

validation_generator = validation_datagen.flow_from_directory(
    validation,
    target_size=(96, 96),
    class_mode='categorical',
    color_mode='grayscale'  # Ajustar según el modo de color de tus imágenes
)

# Indices de entrada y salida del modelo
tflite_model_file = pathlib.Path("simpler_model.tflite")
# tflite_model_file.write_bytes(tflite_model)

# # Inicializa el intérprete de TFLite
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# Esto para el testeo de abajo
test_batches = validation_generator
print(f"Number of images in the test batch: {test_batches.n}")

# Indices de entrada y salida del modelo
# tflite_model_file = 'saved_model/model_t.tflite' # Change the filename here for Model 2 and 3
# interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
# interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

print("Input type: ", input_details[0]['dtype'])
print("Output type: ", output_details[0]['dtype'])

# Añade un parámetro para el índice de la imagen
def test_image(index):
    img, label = next(test_batches)
    single_image = img[index:index+1]
    values = []
    for i in range(96):
        for j in range(96):
            # Añadir cada valor a la lista
            values.append(', '.join(map(str, single_image[0][i][j])))
    # Imprimir todos los valores en una sola línea
    print(', '.join(values))
    interpreter.set_tensor(input_index, single_image)

    image_bytes = single_image.tobytes()
    with open("project_imgs/imageX", "wb") as file:
        file.write(image_bytes)
    display_image_from_bytes(image_bytes, (96, 96))

    interpreter.set_tensor(input_index, single_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    prediction = np.argmax(output)
    true_label = np.argmax(label[index])  # Asegúrate de comparar elementos individuales
    print(f"Para la imagen en el índice {index}, la predicción es {class_names[prediction]} y la etiqueta verdadera es {class_names[true_label]}")
    plt.imshow(np.squeeze(single_image), cmap='gray')
    plt.show()
    return output

# Luego puedes llamar a la función con el índice de la imagen que deseas probar
# output = test_image(0)
# print("Output para la inferencia: ", output)

from PIL import Image

def test_all_images():
    directories = ["project_imgs/cow", "project_imgs/sheep"]
    i = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # add more conditions if there are other image types
                image_path = os.path.join(directory, filename)
                img = Image.open(image_path).convert('L')  # convert image to grayscale
                img_resized = img.resize((96, 96))  # resize the image to 96x96
                single_image = np.array(img_resized).astype('float32') / 255  # convert to float32 and normalize
                single_image = np.expand_dims(single_image, axis=0)  # if your model expects a 4D array
                single_image = np.expand_dims(single_image, axis=-1)  # add an extra dimension for the color channel
                interpreter.set_tensor(input_index, single_image)

                image_bytes = single_image.tobytes()
                with open(f"{directory}/image{i}", "wb") as file:
                    file.write(image_bytes)
                i += 1
                display_image_from_bytes(image_bytes, (96, 96))

                interpreter.set_tensor(input_index, single_image)
                interpreter.invoke()
                output = interpreter.get_tensor(output_index)
                print(f"For the image {filename}, the output is {output}")

test_all_images()