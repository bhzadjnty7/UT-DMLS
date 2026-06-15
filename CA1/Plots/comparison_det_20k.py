import matplotlib.pyplot as plt
import numpy as np

# --- Benchmark Data ---
# Data for 10000x10000 matrix
data_10k = {
    'OpenBLAS': {
        'avg_time': 3.5179,
        'avg_gflops': 189.65,
        'avg_cpu_usage': 8.7
    },
    'MKL': {
        'avg_time': 2.4239,
        'avg_gflops': 275.07,
        'avg_cpu_usage': 4.7
    }
}

# Data for 20000x20000 matrix
data_20k = {
    'OpenBLAS': {
        'avg_time': 22.9118,
        'avg_gflops': 232.93,
        'avg_cpu_usage': 8.8
    },
    'MKL': {
        'avg_time': 16.3350,
        'avg_gflops': 326.50,
        'avg_cpu_usage': 7.6
    }
}

# BLAS labels and colors for consistency
labels = list(data_10k.keys()) # ['OpenBLAS', 'MKL']
colors = ['#1f77b4', '#ff7f0e'] # Blue for OpenBLAS, Orange for MKL

# --- Plot Set 1: Comparison for 10000x10000 Matrix ---
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('BLAS Performance Comparison for 10000x10000 Matrix', fontsize=16)

# Plot Average Execution Time
times_10k = [data_10k[lbl]['avg_time'] for lbl in labels]
bars_time_10k = axes1[0].bar(labels, times_10k, color=colors)
axes1[0].set_title('Average Execution Time (seconds)')
axes1[0].set_ylabel('Time (seconds)')
axes1[0].set_ylim(bottom=0)
for bar in bars_time_10k:
    yval = bar.get_height()
    axes1[0].text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

# Plot Average GFLOPS Performance
gflops_10k = [data_10k[lbl]['avg_gflops'] for lbl in labels]
bars_gflops_10k = axes1[1].bar(labels, gflops_10k, color=colors)
axes1[1].set_title('Average Performance (GFLOPS)')
axes1[1].set_ylabel('GFLOPS')
axes1[1].set_ylim(bottom=0)
for bar in bars_gflops_10k:
    yval = bar.get_height()
    axes1[1].text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}', ha='center', va='bottom')

# Plot Average CPU Usage
cpu_10k = [data_10k[lbl]['avg_cpu_usage'] for lbl in labels]
bars_cpu_10k = axes1[2].bar(labels, cpu_10k, color=colors)
axes1[2].set_title('Average CPU Usage (%)')
axes1[2].set_ylabel('CPU Usage (%)')
axes1[2].set_ylim(bottom=0, top=max(cpu_10k) * 1.5) # Adjust y-limit for better visibility
for bar in bars_cpu_10k:
    yval = bar.get_height()
    axes1[2].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.savefig('benchmark_10k_comparison.png')
plt.show()

# --- Plot Set 2: Comparison for 20000x20000 Matrix ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('BLAS Performance Comparison for 20000x20000 Matrix', fontsize=16)

# Plot Average Execution Time
times_20k = [data_20k[lbl]['avg_time'] for lbl in labels]
bars_time_20k = axes2[0].bar(labels, times_20k, color=colors)
axes2[0].set_title('Average Execution Time (seconds)')
axes2[0].set_ylabel('Time (seconds)')
axes2[0].set_ylim(bottom=0)
for bar in bars_time_20k:
    yval = bar.get_height()
    axes2[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

# Plot Average GFLOPS Performance
gflops_20k = [data_20k[lbl]['avg_gflops'] for lbl in labels]
bars_gflops_20k = axes2[1].bar(labels, gflops_20k, color=colors)
axes2[1].set_title('Average Performance (GFLOPS)')
axes2[1].set_ylabel('GFLOPS')
axes2[1].set_ylim(bottom=0)
for bar in bars_gflops_20k:
    yval = bar.get_height()
    axes2[1].text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f}', ha='center', va='bottom')

# Plot Average CPU Usage
cpu_20k = [data_20k[lbl]['avg_cpu_usage'] for lbl in labels]
bars_cpu_20k = axes2[2].bar(labels, cpu_20k, color=colors)
axes2[2].set_title('Average CPU Usage (%)')
axes2[2].set_ylabel('CPU Usage (%)')
axes2[2].set_ylim(bottom=0, top=max(cpu_20k) * 1.5) # Adjust y-limit for better visibility
for bar in bars_cpu_20k:
    yval = bar.get_height()
    axes2[2].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('benchmark_20k_comparison.png')
plt.show()

# --- Plot Set 3: Combined Comparison Across Matrix Sizes ---
# This set of plots shows how each BLAS scales with increasing matrix size.

matrix_sizes_labels = ['10000x10000', '20000x20000']
x = np.arange(len(matrix_sizes_labels)) # Label locations
width = 0.35 # Bar width

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('BLAS Performance Scaling Across Matrix Sizes', fontsize=16)

# Combined Plot for Average Execution Time
times_openblas = [data_10k['OpenBLAS']['avg_time'], data_20k['OpenBLAS']['avg_time']]
times_mkl = [data_10k['MKL']['avg_time'], data_20k['MKL']['avg_time']]
rects1_time = axes3[0].bar(x - width/2, times_openblas, width, label='OpenBLAS', color=colors[0])
rects2_time = axes3[0].bar(x + width/2, times_mkl, width, label='MKL', color=colors[1])
axes3[0].set_title('Average Execution Time (seconds)')
axes3[0].set_ylabel('Time (seconds)')
axes3[0].set_xticks(x)
axes3[0].set_xticklabels(matrix_sizes_labels)
axes3[0].legend()
axes3[0].set_ylim(bottom=0)
for rect in rects1_time + rects2_time:
    height = rect.get_height()
    axes3[0].annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Combined Plot for Average GFLOPS Performance
gflops_openblas = [data_10k['OpenBLAS']['avg_gflops'], data_20k['OpenBLAS']['avg_gflops']]
gflops_mkl = [data_10k['MKL']['avg_gflops'], data_20k['MKL']['avg_gflops']]
rects1_gflops = axes3[1].bar(x - width/2, gflops_openblas, width, label='OpenBLAS', color=colors[0])
rects2_gflops = axes3[1].bar(x + width/2, gflops_mkl, width, label='MKL', color=colors[1])
axes3[1].set_title('Average Performance (GFLOPS)')
axes3[1].set_ylabel('GFLOPS')
axes3[1].set_xticks(x)
axes3[1].set_xticklabels(matrix_sizes_labels)
axes3[1].legend()
axes3[1].set_ylim(bottom=0)
for rect in rects1_gflops + rects2_gflops:
    height = rect.get_height()
    axes3[1].annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Combined Plot for Average CPU Usage
cpu_openblas = [data_10k['OpenBLAS']['avg_cpu_usage'], data_20k['OpenBLAS']['avg_cpu_usage']]
cpu_mkl = [data_10k['MKL']['avg_cpu_usage'], data_20k['MKL']['avg_cpu_usage']]
rects1_cpu = axes3[2].bar(x - width/2, cpu_openblas, width, label='OpenBLAS', color=colors[0])
rects2_cpu = axes3[2].bar(x + width/2, cpu_mkl, width, label='MKL', color=colors[1])
axes3[2].set_title('Average CPU Usage (%)')
axes3[2].set_ylabel('CPU Usage (%)')
axes3[2].set_xticks(x)
axes3[2].set_xticklabels(matrix_sizes_labels)
axes3[2].legend()
axes3[2].set_ylim(bottom=0, top=max(max(cpu_openblas), max(cpu_mkl)) * 1.5)
for rect in rects1_cpu + rects2_cpu:
    height = rect.get_height()
    axes3[2].annotate(f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('benchmark_combined_comparison.png')
plt.show()