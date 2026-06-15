import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your experiments
batch_sizes = [16, 32, 64, 128]
total_times = [277.63, 133.39, 129.63, 112.62]
accuracies = [65.49, 66.79, 67.38, 67.21]
memories = [0.60, 0.79, 1.17, 1.93]

def plot_batch_size_analysis():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Training time chart
    ax1.plot(batch_sizes, total_times, marker='o', color='#d62728', linewidth=2)
    ax1.set_title('Training Time vs Batch Size', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Batch Size (Per GPU)')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Display values on the points
    for x, y in zip(batch_sizes, total_times):
        ax1.text(x, y+5, f'{y:.1f}s', ha='center', fontsize=9)

    # 2. Memory usage chart
    ax2.plot(batch_sizes, memories, marker='s', color='#1f77b4', linewidth=2)
    ax2.set_title('GPU Memory Usage vs Batch Size', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Batch Size (Per GPU)')
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    for x, y in zip(batch_sizes, memories):
        ax2.text(x, y+0.1, f'{y:.2f}GB', ha='center', fontsize=9)

    # 3. Accuracy chart
    ax3.plot(batch_sizes, accuracies, marker='^', color='#2ca02c', linewidth=2)
    ax3.set_title('Test Accuracy vs Batch Size', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Batch Size (Per GPU)')
    ax3.set_ylabel('Best Accuracy (%)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    for x, y in zip(batch_sizes, accuracies):
        ax3.text(x, y+0.2, f'{y:.2f}%', ha='center', fontsize=9)

    plt.suptitle('Impact of Batch Size on Distributed Training (2 GPUs)', fontsize=16)
    plt.tight_layout()
    plt.show()

# Execute the plotting function
plot_batch_size_analysis()