import matplotlib.pyplot as plt
import numpy as np

# --- Data from all experiments ---
methods = [
    'Serial (Centralized)', 
    'Federated\n(1 Round, 10 Local Epochs)', 
    'Federated\n(10 Rounds, 1 Local Epoch)'
]
total_times = [0.8533, 26.4442, 25.0988]
accuracies = [65.85, 62.44, 63.08]

# --- Plot 1: Total Execution Time Comparison ---
fig1, ax1 = plt.subplots(figsize=(10, 7))
bars1 = ax1.bar(methods, total_times, color=['#4CAF50', '#FFC107', '#2196F3'])

ax1.set_ylabel('Total Execution Time (seconds)')
ax1.set_title('Comparison of Total Execution Times')
ax1.set_yscale('log') # Use log scale for better visualization due to large difference
ax1.set_ylim(bottom=0.1)

# Add text labels on bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center', size=12)

plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.show()


# --- Plot 2: Final Accuracy Comparison ---
fig2, ax2 = plt.subplots(figsize=(10, 7))
bars2 = ax2.bar(methods, accuracies, color=['#4CAF50', '#FFC107', '#2196F3'])

ax2.set_ylabel('Average Test Accuracy (%)')
ax2.set_title('Comparison of Final Model Accuracies')
ax2.set_ylim(60, 68) # Zoom in on the relevant accuracy range

# Add text labels on bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center', size=12)

plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.show()