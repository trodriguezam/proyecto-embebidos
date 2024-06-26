import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import numpy as np

validation = "project_imgs"
    
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
    color_mode='grayscale',  # Ajustar según el modo de color de tus imágenes
)

test_batches = validation_generator


# def test_image(index):
#     # img, label = next(test_batches)
#     # single_image = img[index:index+1]
#     single_image = index
#     image_bytes = single_image.tobytes()
#     with open(f"project_imgs/other/image{index}", "wb") as file:
#         file.write(image_bytes)
#     display_image_from_bytes(image_bytes, (96, 96))




# Assuming 'validation_datagen' is already defined and configured
# validation_generator = validation_datagen.flow_from_directory(
#     validation,
#     target_size=(96, 96),
#     class_mode='categorical',
#     color_mode='grayscale',  # Adjust according to your images' color mode
#     shuffle=False  # Important to keep the order, if you need to map back to filenames
# )

# Find the index of the "other" class
other_class_index = list(validation_generator.class_indices.keys()).index('other')

def save_other_class_images(batch_size=32, num_images=3):
    saved_images = 0
    for img_batch, label_batch in validation_generator:
        for i in range(img_batch.shape[0]):
            if np.argmax(label_batch[i]) == other_class_index:
                image_bytes = img_batch[i].tobytes()
                with open(f"project_imgs/other/image{saved_images}", "wb") as file:
                    file.write(image_bytes)
                saved_images += 1
                if saved_images >= num_images:
                    return

# Call the function to save images from the "other" class
save_other_class_images(batch_size=32, num_images=3)

for i in range(3):
    with open(f"project_imgs/other/image{i}", "rb") as file:
        image_bytes = file.read()
        display_image_from_bytes(image_bytes, (96, 96))