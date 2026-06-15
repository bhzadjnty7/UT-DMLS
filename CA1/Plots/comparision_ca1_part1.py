import matplotlib.pyplot as plt
import numpy as np

# --- Data from all experiments ---
# Total execution times
methods = ['Serial', 'Block (2 tasks)', 'Block (4 tasks)', 'Cyclic (2 tasks)', 'Cyclic (4 tasks)']
total_times = [22.0950, 17.2639, 10.2117, 11.2210, 5.5928]

# Per-process times for 4-task scenarios
block_4_tasks_times = [1.2517, 3.9179, 6.8383, 10.2084]
cyclic_4_tasks_times = [5.5445, 5.5483, 5.5896, 5.5553]
process_labels = ['Process 0', 'Process 1', 'Process 2', 'Process 3']

# --- Plot 1: Total Execution Time Comparison ---
fig1, ax1 = plt.subplots(figsize=(12, 7))
bars1 = ax1.bar(methods, total_times, color=['red', 'orange', 'gold', 'skyblue', 'dodgerblue'])

ax1.set_ylabel('Total Execution Time (seconds)')
ax1.set_title('Comparison of Total Execution Times for Pi Estimation Methods')
ax1.set_ylim(0, max(total_times) * 1.1)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}s', va='bottom', ha='center')

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()


# --- Plot 2: Per-Process Time Comparison (Load Balancing) for 4 tasks ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(len(process_labels))
width = 0.35

rects1 = ax2.bar(x - width/2, block_4_tasks_times, width, label='Block Distribution', color='orange')
rects2 = ax2.bar(x + width/2, cyclic_4_tasks_times, width, label='Cyclic Distribution', color='dodgerblue')

ax2.set_ylabel('Per-Process Execution Time (seconds)')
ax2.set_title('Load Balancing Comparison (4 Processes)')
ax2.set_xticks(x)
ax2.set_xticklabels(process_labels)
ax2.legend()

# Add labels on bars
ax2.bar_label(rects1, padding=3, fmt='%.2fs')
ax2.bar_label(rects2, padding=3, fmt='%.2fs')

plt.tight_layout()
plt.show()