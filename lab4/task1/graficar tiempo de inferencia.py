# Tiempos proporcionados en milisegundos
total_time_measured = 54465.905
inference_time = 54483  # Inference time

quantize_time = 5.768
dequantize_time = 0.016
reshape_time = 0.109
relu_time = 0.0
maxpool_time = 215.751
conv2d_time = 54018.900
fully_connected_time = 224.491
logistic_time = 0.091

# Sumar los tiempos de los subtasks
total_subtasks_time = (quantize_time + dequantize_time + reshape_time + relu_time +
                       maxpool_time + conv2d_time + fully_connected_time + logistic_time)

# Comparar tiempos
print(f"Tiempo total medido: {total_time_measured} ms")
print(f"Tiempo total de inferencia: {inference_time} ms")
print(f"Suma de tiempos de los subtasks: {total_subtasks_time} ms")

# Verificar si la suma de los subtasks es similar al tiempo total medido
print(f"Diferencia: {abs(total_subtasks_time - total_time_measured)} ms")

import matplotlib.pyplot as plt
import numpy as np
# Datos para el gráfico
tasks = ['Quantize', 'Dequantize', 'Reshape', 'Relu', 'Maxpool', 'Conv2D', 'FullyConnected', 'Logistic']
times = [quantize_time, dequantize_time, reshape_time, relu_time, maxpool_time, conv2d_time, fully_connected_time, logistic_time]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(tasks, times, color='blue')
plt.xlabel('Subtasks')
plt.ylabel('Time (ms)')
plt.title('Time taken by each subtask')
output_path = 'tiempo_de_inferencia.png'
plt.savefig(output_path)
plt.show()

# Identificar el "bottleneck"
bottleneck_task = tasks[np.argmax(times)]
bottleneck_time = max(times)
print(f"El 'bottleneck' es: {bottleneck_task} con un tiempo de {bottleneck_time} ms")
