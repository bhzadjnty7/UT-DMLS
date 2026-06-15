import matplotlib.pyplot as plt
import numpy as np

# --- Data from all Logistic Regression experiments ---
scenarios = [
    'Serial (Baseline)', 
    'FedAvg: 1 Round, 10 Epochs', 
    'FedAvg: 10 Rounds, 1 Epoch',
    'FedAvg: Malicious Client'
]

# Total training times
training_times = [
    0.8533,    # Serial
    26.1051,   # Scenario 1
    23.8758,   # Scenario 2
    21.8127    # Malicious
]

# Final average accuracies
accuracies = [
    65.85,     # Serial
    62.44,     # Scenario 1
    63.08,     # Scenario 2
    60.60      # Malicious
]


# --- Plot 1: Training Time Comparison ---
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 7))

bars1 = ax1.bar(scenarios, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_ylabel('Total Training Time (seconds)')
ax1.set_title('Training Time Comparison: Serial vs. Federated Learning Scenarios')
ax1.set_yscale('log') # Use log scale due to large difference between serial and parallel

# Add text labels on bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center', fontsize=10)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()


# --- Plot 2: Final Accuracy Comparison ---
fig2, ax2 = plt.subplots(figsize=(12, 7))

bars2 = ax2.bar(scenarios, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_ylabel('Final Average Test Accuracy (%)')
ax2.set_title('Final Accuracy Comparison: Serial vs. Federated Learning Scenarios')
ax2.set_ylim(55, 70) # Set y-axis limit to better visualize differences

# Add text labels on bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center', fontsize=10)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()