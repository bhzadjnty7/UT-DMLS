#!/usr/bin/env python3
"""
Benchmark: Matrix Inverse Computation (10000x10000)
DMLS - CA1 - Part 3.2

This script benchmarks the matrix inversion of a square matrix
using different BLAS backends (OpenBLAS, MKL, etc.)
"""

import numpy as np
import time
import os
import psutil
import platform
import sys

# Import helper functions (same as determinant benchmark)
# ... [Copy get_system_info, get_blas_info, etc. from bench_det_10k.py] ...
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

def benchmark_inverse(size=10000, n_runs=3):
    """
    Benchmark matrix inversion for a square matrix
    
    Matrix inversion via LU decomposition has complexity of
    approximately (8/3) * n^3 floating point operations:
    - LU decomposition: (2/3) * n^3
    - Forward/backward substitution: 2 * n^3
    
    Args:
        size (int): Size of the square matrix (size x size)
        n_runs (int): Number of benchmark runs for averaging
    
    Returns:
        dict: Benchmark results including timing and performance metrics
    """
    print("=" * 70)
    print(f"Benchmark: Matrix Inverse ({size}x{size})")
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
    
    # Memory requirement (need space for both A and A_inv)
    memory_required = 2 * size * size * 8 / (1024**3)  # GB
    print(f"\n💾 Memory Required: {memory_required:.2f} GB")
    
    print("\n" + "-" * 70)
    print("Starting benchmark...")
    print("-" * 70)
    
    # Storage for results
    times = []
    gflops_list = []
    cpu_usage_list = []
    errors = []
    
    for run in range(n_runs):
        print(f"\n🔄 Run {run + 1}/{n_runs}:")
        
        # Generate random invertible matrix
        print("  📝 Generating random invertible matrix...")
        np.random.seed(42 + run)
        A = np.random.randn(size, size).astype(np.float64)
        # Add to diagonal to ensure invertibility and good conditioning
        A = A @ A.T + np.eye(size) * 0.1
        
        print(f"  ✓ Matrix generated ({A.nbytes / (1024**2):.2f} MB)")
        
        # Check condition number (only for smaller matrices)
        if size <= 1000:
            cond = np.linalg.cond(A)
            print(f"  ✓ Condition number: {cond:.2e}")
        
        # Warm-up run (only for first iteration)
        if run == 0:
            print("  🔥 Warming up CPU...")
            _ = np.linalg.inv(A[:100, :100])
            time.sleep(0.5)
        
        # Start CPU monitoring
        cpu_before = psutil.cpu_percent(interval=None, percpu=True)
        
        # Compute matrix inverse
        print("  ⚙️  Computing matrix inverse...")
        start_time = time.perf_counter()
        A_inv = np.linalg.inv(A)
        end_time = time.perf_counter()
        
        # Get CPU usage after computation
        cpu_usage = get_cpu_usage_detailed()
        cpu_usage_list.append(cpu_usage)
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Calculate GFLOPS
        # Matrix inversion via LU: approximately (8/3) * n^3 FLOPs
        flops = (8.0 / 3.0) * (size ** 3)
        gflops = (flops / elapsed) / 1e9
        gflops_list.append(gflops)
        
        # Verify correctness (sample check for large matrices)
        print("  🔍 Verifying correctness...")
        if size <= 1000:
            # Full verification for small matrices
            identity = A @ A_inv
            error = np.linalg.norm(identity - np.eye(size), 'fro') / size
        else:
            # Sample verification for large matrices
            sample_size = min(1000, size)
            identity_sample = A[:sample_size, :] @ A_inv[:, :sample_size]
            error = np.linalg.norm(identity_sample - np.eye(sample_size), 'fro') / sample_size
        
        errors.append(error)
        
        # Display results for this run
        print(f"  ✓ Time: {elapsed:.4f} seconds")
        print(f"  ✓ Performance: {gflops:.2f} GFLOPS")
        print(f"  ✓ Verification error: {error:.6e}")
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
    
    avg_error = np.mean(errors)
    
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
    print(f"\n✅ Accuracy:")
    print(f"  Average verification error: {avg_error:.6e}")
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
        'avg_error': avg_error,
        'sys_info': sys_info,
        'thread_info': thread_info
    }

if __name__ == "__main__":
    # Parse command line arguments
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Run benchmark
    results = benchmark_inverse(size=size, n_runs=n_runs)
    
    # Save results to JSON
    import json
    output_file = f'bench_inv_{size//1000}k_results.json'
    
    results_serializable = results.copy()
    results_serializable['times'] = [float(x) for x in results['times']]
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")