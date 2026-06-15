#!/usr/bin/env python3
"""
Benchmark: Matrix Determinant (10000x10000)
DMLS - CA1 - Part 3.1

This script benchmarks the computation of matrix determinant
using different BLAS backends (OpenBLAS, MKL, etc.)
"""

import numpy as np
import time
import os
import psutil
import platform
import json

def get_system_info():
    """
    Get system information including CPU, RAM, and Python version
    
    Returns:
        dict: System information
    """
    info = {
        'CPU': platform.processor(),
        'CPU Cores (Physical)': psutil.cpu_count(logical=False),
        'CPU Threads (Logical)': psutil.cpu_count(logical=True),
        'RAM': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'Python Version': platform.python_version(),
        'NumPy Version': np.__version__,
        'Platform': platform.system(),
        'Architecture': platform.machine()
    }
    return info

def get_blas_info():
    """
    Get BLAS/LAPACK configuration from NumPy
    
    Returns:
        str: BLAS configuration information
    """
    # Capture np.show_config() output
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    np.show_config()
    
    sys.stdout = old_stdout
    config = buffer.getvalue()
    
    return config

def get_thread_config():
    """
    Get thread configuration from environment variables
    
    Returns:
        dict: Thread configuration for different BLAS libraries
    """
    thread_info = {
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'Not set (auto)'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'Not set (auto)'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'Not set (auto)'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', 'Not set (auto)')
    }
    return thread_info

def get_cpu_usage_stats(cpu_percents):
    """
    Calculate CPU usage statistics
    
    Args:
        cpu_percents: List of CPU usage percentages per core
    
    Returns:
        dict: CPU usage statistics
    """
    return {
        'mean': np.mean(cpu_percents),
        'max': np.max(cpu_percents),
        'min': np.min(cpu_percents),
        'active_cores': sum(1 for x in cpu_percents if x > 10),
        'total_cores': len(cpu_percents)
    }

def calculate_flops_determinant(size):
    """
    Calculate theoretical FLOPs for determinant computation
    
    Determinant is computed using LU decomposition which requires
    approximately (2/3) * n^3 floating point operations
    
    Args:
        size: Matrix size (n x n)
    
    Returns:
        float: Number of FLOPs
    """
    # LU decomposition complexity: O((2/3) * n^3)
    flops = (2.0 / 3.0) * (size ** 3)
    return flops

def benchmark_determinant(size=10000, n_runs=3):
    """
    Benchmark matrix determinant computation
    
    Args:
        size: Matrix size (default: 10000)
        n_runs: Number of runs for averaging (default: 3)
    
    Returns:
        dict: Benchmark results including timing and performance metrics
    """
    print("=" * 70)
    print(f"Benchmark: Matrix Determinant ({size}x{size})")
    print("=" * 70)
    
    # Display system information
    print("\n📊 System Information:")
    print("-" * 70)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Display BLAS configuration
    print("\n🔧 BLAS Configuration:")
    print("-" * 70)
    blas_config = get_blas_info()
    print(blas_config)
    
    # Display thread configuration
    print("\n🧵 Thread Configuration:")
    print("-" * 70)
    thread_config = get_thread_config()
    for key, value in thread_config.items():
        print(f"  {key}: {value}")
    
    # Calculate memory requirements
    memory_mb = (size * size * 8) / (1024 ** 2)  # 8 bytes per float64
    print(f"\n💾 Memory Requirements:")
    print(f"  Matrix size: {memory_mb:.2f} MB")
    
    print("\n" + "-" * 70)
    print("Starting benchmark...")
    print("-" * 70)
    
    # Storage for results
    times = []
    gflops_list = []
    cpu_stats_list = []
    
    for run in range(n_runs):
        print(f"\n🔄 Run {run + 1}/{n_runs}:")
        
        # Generate random matrix
        print("  → Generating random matrix...")
        np.random.seed(42 + run)  # Reproducible random seed
        A = np.random.rand(size, size).astype(np.float64)
        
        # Warm-up run (only on first iteration)
        if run == 0:
            print("  → Warming up CPU...")
            _ = np.linalg.det(A[:100, :100])
            time.sleep(1)  # Let CPU settle
        
        # Start CPU monitoring
        psutil.cpu_percent(interval=None, percpu=True)  # Reset counters
        
        # Compute determinant
        print("  → Computing determinant...")
        start_time = time.perf_counter()
        det = np.linalg.det(A)
        end_time = time.perf_counter()
        
        # Get CPU usage after computation
        cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_stats = get_cpu_usage_stats(cpu_percents)
        cpu_stats_list.append(cpu_stats)
        
        # Calculate elapsed time
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Calculate GFLOPS (Giga Floating Point Operations Per Second)
        flops = calculate_flops_determinant(size)
        gflops = (flops / elapsed) / 1e9
        gflops_list.append(gflops)
        
        # Display results for this run
        print(f"  ✓ Time: {elapsed:.4f} seconds")
        print(f"  ✓ Performance: {gflops:.2f} GFLOPS")
        print(f"  ✓ Determinant: {det:.6e}")
        print(f"  ✓ CPU Usage: {cpu_stats['mean']:.1f}% (avg), "
              f"{cpu_stats['max']:.1f}% (max)")
        print(f"  ✓ Active Cores: {cpu_stats['active_cores']}/{cpu_stats['total_cores']}")
    
    # Calculate final statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    avg_gflops = np.mean(gflops_list)
    max_gflops = np.max(gflops_list)
    min_gflops = np.min(gflops_list)
    
    avg_cpu_usage = np.mean([s['mean'] for s in cpu_stats_list])
    max_cpu_usage = np.max([s['max'] for s in cpu_stats_list])
    
    # Display final results
    print("\n" + "=" * 70)
    print("📈 FINAL RESULTS")
    print("=" * 70)
    print(f"Matrix Size: {size}x{size}")
    print(f"Number of runs: {n_runs}")
    print(f"\n⏱️  Timing Statistics:")
    print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"\n🚀 Performance:")
    print(f"  Average: {avg_gflops:.2f} GFLOPS")
    print(f"  Peak: {max_gflops:.2f} GFLOPS")
    print(f"  Min: {min_gflops:.2f} GFLOPS")
    print(f"\n💻 CPU Usage:")
    print(f"  Average: {avg_cpu_usage:.1f}%")
    print(f"  Peak: {max_cpu_usage:.1f}%")
    print("=" * 70)
    
    # Prepare results dictionary
    results = {
        'benchmark_type': 'determinant',
        'matrix_size': size,
        'n_runs': n_runs,
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'gflops': gflops_list,
        'avg_gflops': avg_gflops,
        'max_gflops': max_gflops,
        'min_gflops': min_gflops,
        'cpu_stats': cpu_stats_list,
        'avg_cpu_usage': avg_cpu_usage,
        'max_cpu_usage': max_cpu_usage,
        'sys_info': sys_info,
        'thread_config': thread_config,
        'blas_config': blas_config
    }
    
    return results

if __name__ == "__main__":
    import sys
    
    # Get matrix size from command line argument (default: 10000)
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    
    # Run benchmark
    results = benchmark_determinant(size=size, n_runs=3)
    
    # Save results to JSON file
    output_file = f'bench_det_{size//1000}k_results.json'
    
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = results.copy()
    results_serializable['times'] = [float(x) for x in results['times']]
    results_serializable['gflops'] = [float(x) for x in results['gflops']]
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")