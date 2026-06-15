import matplotlib.pyplot as plt
import numpy as np

# Data setup
scenarios = ['SMSC (1N/1P)', 'SMMC (1N/2P)', 'MMSC (2N/1P)', 'MMMC (2N/4P)']

# Q1 Times (Native PyTorch)
q1_times = [1244.78, 834.85, 1426.17, 823.59]

# Q2 Times (Hugging Face Accelerate - Corrected SMMC)
q2_times = [1205.73, 856.18, 1499.99, 863.50]

x = np.arange(len(scenarios))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - width/2, q1_times, width, label='Q1: Native PyTorch', color='#2980b9', alpha=0.9)
rects2 = ax.bar(x + width/2, q2_times, width, label='Q2: HF Accelerate', color='#e74c3c', alpha=0.9)

# Labels
ax.set_ylabel('Training Time (Seconds)', fontweight='bold')
ax.set_title('Performance Comparison: Native PyTorch vs. Accelerate', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontweight='bold')
ax.legend()

# Add values on top
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()