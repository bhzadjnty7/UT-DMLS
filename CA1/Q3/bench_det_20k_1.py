#!/usr/bin/env python3
"""
Benchmark: Matrix Determinant Computation (20000x20000)
DMLS - CA1 - Part 3.1

This script benchmarks the determinant computation of a large square matrix.
Note: This requires significant memory (approximately 3.2 GB).
"""

import numpy as np
import time
import os
import psutil
import platform
import sys

def get_system_info():
    """
    Get system information including CPU, memory, and software versions
    
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
        'Platform': platform.platform()
    }
    return info

def get_blas_info():
    """
    Get BLAS backend configuration
    
    Returns:
        str: BLAS configuration details
    """
    try:
        config = np.__config__.show()
        return config
    except:
        return "BLAS info not available"

def get_thread_config():
    """
    Get thread configuration from environment variables
    
    Returns:
        dict: Thread configuration for different BLAS backends
    """
    thread_info = {
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'Not set (using default)'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'Not set (using default)'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'Not set (using default)'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', 'Not set (using default)')
    }
    return thread_info

def get_cpu_usage_detailed():
    """
    Get detailed CPU usage per core
    
    Returns:
        dict: CPU usage statistics
    """
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    return {
        'per_core': cpu_percent,
        'average': np.mean(cpu_percent),
        'max': np.max(cpu_percent),
        'active_cores': sum(1 for x in cpu_percent if x > 10)
    }

def benchmark_determinant(size=10000, n_runs=3):
    """
    Benchmark determinant computation for a square matrix
    
    The determinant is computed using LU decomposition which has
    complexity of approximately (2/3) * n^3 floating point operations.
    
    Args:
        size (int): Size of the square matrix (size x size)
        n_runs (int): Number of benchmark runs for averaging
    
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
    print(get_blas_info())
    
    # Display thread configuration
    print("\n🧵 Thread Configuration:")
    print("-" * 70)
    thread_info = get_thread_config()
    for key, value in thread_info.items():
        print(f"  {key}: {value}")
    
    # Memory requirement
    memory_required = size * size * 8 / (1024**3)  # GB
    print(f"\n💾 Memory Required: {memory_required:.2f} GB")
    
    print("\n" + "-" * 70)
    print("Starting benchmark...")
    print("-" * 70)
    
    # Storage for results
    times = []
    gflops_list = []
    cpu_usage_list = []
    
    for run in range(n_runs):
        print(f"\n🔄 Run {run + 1}/{n_runs}:")
        
        # Generate random matrix
        print("  📝 Generating random matrix...")
        np.random.seed(42 + run)  # Different seed for each run
        A = np.random.randn(size, size).astype(np.float64)
        
        print(f"  ✓ Matrix generated ({A.nbytes / (1024**2):.2f} MB)")
        
        # Warm-up run (only for first iteration)
        if run == 0:
            print("  🔥 Warming up CPU...")
            _ = np.linalg.det(A[:100, :100])
            time.sleep(0.5)  # Let CPU settle
        
        # Start CPU monitoring
        cpu_before = psutil.cpu_percent(interval=None, percpu=True)
        
        # Compute determinant
        print("  ⚙️  Computing determinant...")
        start_time = time.perf_counter()
        det = np.linalg.det(A)
        end_time = time.perf_counter()
        
        # Get CPU usage after computation
        cpu_usage = get_cpu_usage_detailed()
        cpu_usage_list.append(cpu_usage)
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Calculate GFLOPS
        # Determinant via LU decomposition: approximately (2/3) * n^3 FLOPs
        flops = (2.0 / 3.0) * (size ** 3)
        gflops = (flops / elapsed) / 1e9
        gflops_list.append(gflops)
        
        # Display results for this run
        print(f"  ✓ Time: {elapsed:.4f} seconds")
        print(f"  ✓ Performance: {gflops:.2f} GFLOPS")
        print(f"  ✓ Determinant value: {det:.6e}")
        print(f"  ✓ CPU Usage: {cpu_usage['average']:.1f}% (avg), "
              f"{cpu_usage['max']:.1f}% (max)")
        print(f"  ✓ Active Cores: {cpu_usage['active_cores']}/"
              f"{len(cpu_usage['per_core'])}")
    
    # Calculate final statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    avg_gflops = np.mean(gflops_list)
    max_gflops = np.max(gflops_list)
    
    avg_cpu_usage = np.mean([x['average'] for x in cpu_usage_list])
    avg_active_cores = np.mean([x['active_cores'] for x in cpu_usage_list])
    
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
    print(f"\n💻 CPU Utilization:")
    print(f"  Average CPU usage: {avg_cpu_usage:.1f}%")
    print(f"  Average active cores: {avg_active_cores:.1f}")
    print("=" * 70)
    
    return {
        'size': size,
        'n_runs': n_runs,
        'times': times,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_gflops': avg_gflops,
        'max_gflops': max_gflops,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_active_cores': avg_active_cores,
        'sys_info': sys_info,
        'thread_info': thread_info
    }

if __name__ == "__main__":
    # Default size is 20000
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    required_memory = size * size * 8 / (1024**3)  # GB
    
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Required memory: {required_memory:.2f} GB")
    
    if required_memory > available_memory * 0.8:
        print("\n⚠️  WARNING: Insufficient memory!")
        print("Consider using a smaller matrix size or closing other applications.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
    
    # Run benchmark
    results = benchmark_determinant(size=size, n_runs=n_runs)
    
    # Save results
    import json
    output_file = f'bench_det_{size//1000}k_results.json'
    
    results_serializable = results.copy()
    results_serializable['times'] = [float(x) for x in results['times']]
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")