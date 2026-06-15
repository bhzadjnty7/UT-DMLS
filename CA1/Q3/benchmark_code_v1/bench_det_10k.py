# bench_det_10k.py
import numpy as np
import time
import platform
import sys

def benchmark_determinant(matrix_size, n_iterations=5):
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
        np.random.seed(42 + i)  # Different seed for each iteration
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
    
    # Check which BLAS library is being used
    try:
        import numpy.distutils.system_info as sysinfo
        blas_info = sysinfo.get_info('blas_opt')
        lapack_info = sysinfo.get_info('lapack_opt')
        
        print("\nBLAS Library Information:")
        if blas_info:
            print(f"BLAS: {blas_info}")
        else:
            print("BLAS: Information not available")
            
        print("\nLAPACK Library Information:")
        if lapack_info:
            print(f"LAPACK: {lapack_info}")
        else:
            print("LAPACK: Information not available")
    except:
        print("\nCould not retrieve BLAS/LAPACK information")
    
    print("=" * 70)

def main():
    # Print system information
    print_system_info()
    
    print("\n")
    print("=" * 70)
    print("DETERMINANT BENCHMARK - 10000x10000 Matrix")
    print("=" * 70)
    print("\n")
    
    # Benchmark for 10000x10000 matrix
    matrix_size = 10000
    n_iterations = 5
    
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