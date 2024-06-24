import numpy as np
import matplotlib.pyplot as plt

# Tiempos proporcionados
total_time = 54465.905
conv2d_time = 54018.900
fully_connected_time = 224.491

# Calcular el tiempo paralelizable
parallelizable_time = conv2d_time + fully_connected_time

# Calcular la fracción paralelizable P
P = parallelizable_time / total_time

# Definir el rango de núcleos
cores = np.arange(1, 17)  # De 1 a 17 núcleos

# Calcular la aceleración para cada número de núcleos usando la Ley de Amdahl
acceleration = 1 / ((1 - P) + (P / cores))

# Graficar el número de núcleos vs aceleración
plt.figure(figsize=(10, 6))
plt.plot(cores, acceleration, marker='o')
plt.xlabel('Number of Cores')
plt.ylabel('Acceleration')
plt.title('Number of Cores vs Acceleration')
plt.grid(True)
output_path = 'aceleracion_vs_cores.png'
plt.savefig(output_path)
plt.show()

# Encontrar el número de núcleos con la máxima aceleración
max_acceleration = np.max(acceleration)
optimal_cores = cores[np.argmax(acceleration)]

print(f"Máxima aceleración: {max_acceleration}")
print(f"Número óptimo de núcleos: {optimal_cores}")
