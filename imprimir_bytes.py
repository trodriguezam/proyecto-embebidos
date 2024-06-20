import matplotlib.pyplot as plt
import numpy as np

def display_image_from_bytes(image_bytes, shape):
    # Convertir los bytes a una matriz numpy
    image = np.frombuffer(image_bytes, dtype=np.float32).reshape(shape)

    # Visualizar la imagen
    plt.imshow(image, cmap='gray')
    plt.show()

with open("sample_images/image8", "rb") as file:
        image_bytes = file.read()
display_image_from_bytes(image_bytes, (96, 96))