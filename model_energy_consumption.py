import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Your original data
ops_energy = {
    'Quantize': 0.000711,
    'Dequantize': 0.00000195,
    'Reshape': 0.000013,
    'Relu': 0.000,
    'MaxPool': 0.02632,
    'Conv2D': 6.5903,
    'FullyConnected': 0.027387,
    'Logistic': 0.0000023,
    'Total': 6.64484
}

colors = plt.cm.viridis(np.linspace(0, 1, len(ops_energy)))  # Generate colors

# Plotting the bar chart
plt.figure(figsize=(10, 5))
for (key, value), color in zip(ops_energy.items(), colors):
    plt.xlabel('Operación')
    plt.ylabel('Energía (J)')
    plt.bar(key, value, color=color)  # Use the color for each bar
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Creating a new figure for the legend
plt.figure(figsize=(5, 3))
legend_handles = [mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                                markersize=10, label=f'{key}: {value}') for (key, value), color in zip(ops_energy.items(), colors)]
plt.legend(handles=legend_handles, loc='center', fontsize=8)
plt.axis('off')  # Hide the axes
plt.show()