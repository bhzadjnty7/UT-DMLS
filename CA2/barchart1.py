import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your notebook outputs (Epoch 5 averages/finals)
scenarios = ['cuDNN', 'Fused (Custom)', 'No-cuDNN', 'CPU']

# --- Scenario 1: 2 Layers ---
time_total_2 = [7.90, 8.00, 19.43, 12.10]
time_fwd_2   = [0.60, 0.50, 7.34, 2.86]
time_bwd_2   = [2.35, 3.11, 11.40, 7.58]

# --- Scenario 2: 10 Layers ---
time_total_10 = [9.92, 10.18, 77.91, 40.03]
time_fwd_10   = [2.08, 1.29, 26.01, 7.93]
time_bwd_10   = [5.48, 7.12, 51.15, 31.31]

def plot_benchmark(title, total, fwd, bwd):
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, total, width, label='Total Time', color='#1f77b4')
    rects2 = ax.bar(x, fwd, width, label='Forward Time', color='#2ca02c')
    rects3 = ax.bar(x + width, bwd, width, label='Backward Time', color='#ff7f0e')

    ax.set_ylabel('Time (Seconds per Epoch)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    plt.show()

# Plotting
plot_benchmark('Performance Comparison (2 Conv Layers)', time_total_2, time_fwd_2, time_bwd_2)
plot_benchmark('Performance Comparison (10 Conv Layers)', time_total_10, time_fwd_10, time_bwd_10)
