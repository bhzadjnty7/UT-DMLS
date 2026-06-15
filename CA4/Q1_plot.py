import matplotlib.pyplot as plt
import numpy as np

# Data from the logs provided
scenarios = ['SMSC (1N/1P)', 'SMMC (1N/2P)', 'MMSC (2N/2P)', 'MMMC (2N/4P)', 'No Accum (2N/4P)']
times = [1244.78, 834.85, 1426.17, 823.59, 580.59]
accuracies = [85.64, 86.00, 85.76, 86.39, 86.39]

x = np.arange(len(scenarios))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting Time
color = 'tab:blue'
ax1.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Time (seconds)', color=color, fontsize=12, fontweight='bold')
bars = ax1.bar(x, times, width, color=color, alpha=0.7, label='Time')
ax1.tick_params(axis='y', labelcolor=color)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}s',
             ha='center', va='bottom', color='black', fontweight='bold')

# Plotting Accuracy
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Test Accuracy (%)', color=color, fontsize=12, fontweight='bold')
line = ax2.plot(x, accuracies, color=color, marker='o', linewidth=3, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(84, 87) # Set limit to see differences better

# Add values for points
for i, txt in enumerate(accuracies):
    ax2.text(x[i], txt + 0.1, f'{txt}%', ha='center', color='red', fontweight='bold')

plt.title('Performance Comparison: Time vs Accuracy', fontsize=14, fontweight='bold')
plt.xticks(x, scenarios)
plt.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
plt.show()