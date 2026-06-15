import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Batch 32', 'Batch 128']
nccl_times = [179.61, 113.37]
gloo_times = [244.21, 132.57]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, nccl_times, width, label='NCCL', color='#1f77b4')
rects2 = ax.bar(x + width/2, gloo_times, width, label='Gloo', color='#ff7f0e')

ax.set_ylabel('Training Time (Seconds)')
ax.set_title('Backend Comparison: NCCL vs Gloo')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on top of the bars
ax.bar_label(rects1, padding=3, fmt='%.1f s')
ax.bar_label(rects2, padding=3, fmt='%.1f s')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Extracted data
batch_sizes = [16, 32, 64, 128]
times = [277.63, 133.39, 129.63, 112.62]
memories = [0.60, 0.79, 1.17, 1.93]
accuracies = [65.49, 66.79, 67.38, 67.21]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary axis: time
color = 'tab:red'
ax1.set_xlabel('Batch Size (Per GPU)')
ax1.set_ylabel('Total Training Time (s)', color=color, fontweight='bold')
ax1.plot(batch_sizes, times, marker='o', color=color, linewidth=2, label='Time')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Secondary axis: memory
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('GPU Memory Usage (GB)', color=color, fontweight='bold')
ax2.plot(batch_sizes, memories, marker='s', linestyle='--', color=color, linewidth=2, label='Memory')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Trade-off: Training Time vs. Memory Usage', fontsize=14)
plt.xticks(batch_sizes)
fig.tight_layout()
plt.show()

# Separate plot for accuracy
plt.figure(figsize=(8, 4))
plt.plot(batch_sizes, accuracies, marker='^', color='green', linewidth=2)
plt.xlabel('Batch Size')
plt.ylabel('Test Accuracy (%)')
plt.title('Effect of Batch Size on Accuracy')
plt.grid(True, alpha=0.3)
plt.xticks(batch_sizes)
plt.show()