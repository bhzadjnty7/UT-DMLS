import matplotlib.pyplot as plt
import numpy as np

# --- Data from your Matrix Inverse benchmark (10000x10000) ---
blas_backends = ['OpenBLAS', 'Intel MKL']

# Performance metrics
avg_times = [10.4445, 8.8542]      # Average time in seconds
avg_gflops = [255.35, 303.00]      # Average GFLOPS
avg_cpu_usage = [10.2, 6.9]       # Average CPU usage in percent

# --- Create plots ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('BLAS Performance Comparison: Matrix Inverse (10000x10000)', fontsize=16)

colors = ['#1f77b4', '#ff7f0e'] # Blue for OpenBLAS, Orange for MKL

# --- Plot 1: Average Execution Time ---
bars1 = axes[0].bar(blas_backends, avg_times, color=colors)
axes[0].set_title('Average Execution Time', fontsize=14)
axes[0].set_ylabel('Time (seconds)', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} s', ha='center', va='bottom')

# --- Plot 2: Average GFLOPS ---
bars2 = axes[1].bar(blas_backends, avg_gflops, color=colors)
axes[1].set_title('Average Performance (GFLOPS)', fontsize=14)
axes[1].set_ylabel('GFLOPS', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}', ha='center', va='bottom')

# --- Plot 3: Average CPU Usage ---
bars3 = axes[2].bar(blas_backends, avg_cpu_usage, color=colors)
axes[2].set_title('Average CPU Usage', fontsize=14)
axes[2].set_ylabel('CPU Usage (%)', fontsize=12)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)
axes[2].set_ylim(0, max(avg_cpu_usage) * 1.2)
for bar in bars3:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()