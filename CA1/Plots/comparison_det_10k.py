import matplotlib.pyplot as plt
import numpy as np

# Data from your benchmark results
blas_backends = ['OpenBLAS', 'Intel MKL']
avg_times = [3.5179, 2.4239]  # Average time in seconds
avg_gflops = [189.65, 275.07] # Average GFLOPS

# --- Plot 1: Comparison of Average Execution Time ---
plt.figure(figsize=(8, 6))
bars_time = plt.bar(blas_backends, avg_times, color=['skyblue', 'lightcoral'])
plt.xlabel('BLAS Backend', fontsize=12)
plt.ylabel('Average Execution Time (seconds)', fontsize=12)
plt.title('Comparison of Matrix Determinant Computation Time (10000x10000)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(bottom=0) # Ensure y-axis starts from 0

# Add text labels on top of the bars
for bar in bars_time:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# --- Plot 2: Comparison of Average GFLOPS Performance ---
plt.figure(figsize=(8, 6))
bars_gflops = plt.bar(blas_backends, avg_gflops, color=['lightgreen', 'salmon'])
plt.xlabel('BLAS Backend', fontsize=12)
plt.ylabel('Average GFLOPS', fontsize=12)
plt.title('Comparison of Matrix Determinant Computation Performance (10000x10000)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(bottom=0) # Ensure y-axis starts from 0

# Add text labels on top of the bars
for bar in bars_gflops:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()