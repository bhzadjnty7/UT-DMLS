# bench_det_20k.py
import numpy as np
import time
import platform
import sys

def benchmark_determinant(matrix_size, n_iterations=3):
    """
    Benchmark determinant computation for a given matrix size
    
    Args:
        matrix_size: Size of square matrix (N x N)
        n_iterations: Number of iterations for averaging
    
    Returns:
        Average execution time in seconds
    """
    print(f"Benchmarking determinant computation for {matrix_size}x{matrix_size} matrix")
    print(f"Number of iterations: {n_iterations}")
    print("-" * 70)
    
    times = []
    
    for i in range(n_iterations):
        # Generate random matrix
        np.random.seed(42 + i)
        A = np.random.randn(matrix_size, matrix_size)
        
        # Measure time for determinant computation
        start_time = time.time()
        det = np.linalg.det(A)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        print(f"Iteration {i+1}: {elapsed_time:.4f} seconds (det = {det:.4e})")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("-" * 70)
    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Std deviation: {std_time:.4f} seconds")
    print(f"Min time: {min_time:.4f} seconds")
    print(f"Max time: {max_time:.4f} seconds")
    
    return avg_time

def print_system_info():
    """Print system and NumPy configuration information"""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Try to get BLAS/LAPACK info
    try:
        config = np.__config__.show()
        print("\nNumPy Configuration:")
        print(config)
    except:
        pass
    
    print("=" * 70)

def main():
    # Print system information
    print_system_info()
    
    print("\n")
    print("=" * 70)
    print("DETERMINANT BENCHMARK - 20000x20000 Matrix")
    print("=" * 70)
    print("\n")
    
    # Benchmark for 20000x20000 matrix
    matrix_size = 20000
    n_iterations = 3  # Fewer iterations for larger matrix
    
    avg_time = benchmark_determinant(matrix_size, n_iterations)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Average execution time: {avg_time:.4f} seconds")
    print(f"GFLOPS (approx): {(2/3 * matrix_size**3 / avg_time) / 1e9:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()