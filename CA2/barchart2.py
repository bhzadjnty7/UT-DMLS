import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your outputs
scenarios = ['Single GPU', 'Multi GPU (2x DDP)']
train_times = [87.89, 133.39]    # seconds
accuracies = [71.06, 66.79]      # percent
memory_usage = [0.72, 0.79]      # gigabytes

x = np.arange(len(scenarios))
width = 0.5

# --- Chart 1: Total training time ---
plt.figure(figsize=(8, 5))
bars = plt.bar(x, train_times, width, color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Total Training Time (Seconds)')
plt.title('Training Time Comparison (Lower is Better)')
plt.xticks(x, scenarios)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add numbers on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s', ha='center', va='bottom')
plt.show()

# --- Chart 2: Model accuracy ---
plt.figure(figsize=(8, 5))
bars = plt.bar(x, accuracies, width, color=['#2ca02c', '#d62728'])
plt.ylabel('Best Test Accuracy (%)')
plt.title('Accuracy Comparison (Higher is Better)')
plt.xticks(x, scenarios)
plt.ylim(0, 80)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom')
plt.show()

# --- Chart 3: Memory usage ---
plt.figure(figsize=(8, 5))
bars = plt.bar(x, memory_usage, width, color=['#9467bd', '#8c564b'])
plt.ylabel('Peak GPU Memory (GB)')
plt.title('Memory Usage Comparison per GPU')
plt.xticks(x, scenarios)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f} GB', ha='center', va='bottom')
plt.show()