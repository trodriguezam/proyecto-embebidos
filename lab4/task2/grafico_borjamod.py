import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a dictionary with the provided data
data_dict = {
    'Layer type': ['Conv1', 'Pool1', 'Conv2', 'Pool2', 'Conv3', 'Pool3', 'FullyConnected1', 'FullyConnected2', 'FullyConnected3'],
    'N MAC': [1272384, 2262016, 9331200, 2073600, 7746048, 1721344, 1644800, 33024, 256],
    'Total Bytes Procesados': [150592, 176720, 100144, 81000, 43096, 33620, 6656, 384, 129],
    'Operational Intensity': [8.44921377, 12.8, 93.17782393, 25.6, 179.7393726, 51.2, 247.1153846, 86, 1.984496124]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data_dict)

# Define bandwidth and peak performance based on provided frequency
Freq = 240e6  # 240 MHz

peak_throughput_ops = Freq  # Peak throughput in operations per second (Ops/s)
bandwidth_bytes = Freq  # Memory bandwidth in bytes per second

# Define intensity range
intensity = np.linspace(1e-2, 1e3, 100)  # Operational intensity range

# Calculate the roofline bounds
roof_memory_bound = intensity * bandwidth_bytes  # Memory-bound performance
roof_compute_bound = np.ones_like(intensity) * peak_throughput_ops  # Compute-bound performance

# Plotting the Roofline
plt.figure(figsize=(12, 8))

# Plot Roofline model
plt.plot(intensity, roof_memory_bound, label='Memory Bound', color='blue')
plt.plot(intensity, roof_compute_bound, label='Compute Bound', color='red')

# Scatter plot for the tasks with layer labels
for i, row in df.iterrows():
    plt.scatter(row['Operational Intensity'], row['N MAC'], label=row['Layer type'], color='blue')
    plt.annotate(row['Layer type'], (row['Operational Intensity'], row['N MAC']), textcoords="offset points", xytext=(0,10), ha='center')

# Labels and title
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Operational Intensity (FLOPs/Byte)')
plt.ylabel('Performance (FLOPs)')
plt.title('Roofline Plot of Subtasks')
plt.legend()
plt.grid(True, which="both", ls="--")

# Save the plot as an image file
output_path = 'roofline_plot_borja_edition.png'
plt.savefig(output_path)

# Display the plot
plt.show()
