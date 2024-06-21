import numpy as np

# Dimensiones del tensor
batch_size = 1  # Puede ser cualquier número, aquí se asume 1 para el ejemplo
num_elements = 1  # Para el tensor de forma [-1, 1]

# Cálculo del número de operaciones
num_operations = num_elements * batch_size

# Cálculo de bytes procesados
bytes_int8_input = num_elements * batch_size * 1     # int8 ocupa 1 byte
bytes_int8_output = num_elements * batch_size * 1    # int8 ocupa 1 byte

# Resultados
print(f"Número de operaciones: {num_operations}")
print(f"Bytes procesados por el tensor de entrada (int8): {bytes_int8_input}")
print(f"Bytes procesados por el tensor de salida (int8): {bytes_int8_output}")
