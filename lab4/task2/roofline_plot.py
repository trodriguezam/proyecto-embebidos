import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear un diccionario con los datos proporcionados
data_dict = {
    'Layer type': ['conv2d', 'pool', 'fullyconnected', 'quantize', 'dequantize', 'reshape', 'logistics'],
    'N MAC': [36699264, 174804, 3356160, 9216, 1, 4096, 1],
    'Total Bytes Procesados': [293832, 291340, 7169, 46080, 5, 8192, 2],
    'Operational Intensity': [124.8987993, 0.6, 468.1489748, 0.2, 0.2, 0.5, 0.5]
}

# Convertir el diccionario a un DataFrame
df = pd.DataFrame(data_dict)

# Definir el ancho de banda y el rendimiento máximo basado en la frecuencia proporcionada
Freq = 240e6  # 240 MHz

peak_throughput_ops = Freq  # Rendimiento máximo en operaciones por segundo (Ops/s)
bandwidth_bytes = Freq  # Ancho de banda de memoria en bytes por segundo

# Definir el rango de intensidad
intensity = np.linspace(1e-2, 1e3, 100)  # Rango de intensidad operativa

# Calcular los límites del Roofline
roof_memory_bound = intensity * bandwidth_bytes  # Rendimiento limitado por la memoria
roof_compute_bound = np.ones_like(intensity) * peak_throughput_ops  # Rendimiento limitado por el cómputo

# Graficar el Roofline
plt.figure(figsize=(12, 8))

# Graficar el modelo Roofline
plt.plot(intensity, roof_memory_bound, label='Memory Bound', color='blue')
plt.plot(intensity, roof_compute_bound, label='Compute Bound', color='red')

# Graficar los puntos para las tareas con etiquetas de capas
for i, row in df.iterrows():
    plt.scatter(row['Operational Intensity'], row['N MAC'], label=row['Layer type'], color='blue')
    plt.annotate(row['Layer type'], (row['Operational Intensity'], row['N MAC']), textcoords="offset points", xytext=(0,10), ha='center')

# Etiquetas y título
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (FLOPs)')
plt.title('Roofline Plot of Subtasks')
plt.legend()
plt.grid(True, which="both", ls="--")

# Guardar el gráfico como un archivo PNG
output_path = 'roofline_plot.png'
plt.savefig(output_path)

# Mostrar el gráfico
plt.show()

print(f"El gráfico ha sido guardado en {output_path}")
