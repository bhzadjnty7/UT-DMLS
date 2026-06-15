# pi_parallel_v1.py
from mpi4py import MPI
import time
import math

def compute_term(k):
    """
    Compute a single term of the Euler series with computational cost proportional to k
    """
    s = 0
    for i in range(k // 1000):
        s += i * i
    return 1.0 / (k * k)

def estimate_pi_parallel_block(N, comm):
    """
    Parallel implementation using block (contiguous) distribution
    
    Args:
        N: Total number of terms
        comm: MPI communicator
    
    Returns:
        Estimated value of pi (only valid on rank 0)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Calculate block size for each process
    block_size = N // size
    remainder = N % size
    
    # Determine start and end indices for this process
    if rank < remainder:
        start = rank * (block_size + 1) + 1
        end = start + block_size
    else:
        start = rank * block_size + remainder + 1
        end = start + block_size - 1
    
    # Local computation
    local_start_time = time.time()
    local_sum = 0.0
    
    for k in range(start, end + 1):
        local_sum += compute_term(k)
    
    local_end_time = time.time()
    local_time = local_end_time - local_start_time
    
    # Gather all local sums to rank 0
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    
    # Gather timing information
    all_times = comm.gather(local_time, root=0)
    
    if rank == 0:
        pi_estimate = math.sqrt(6.0 * total_sum)
        return pi_estimate, all_times
    else:
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 500000  # Number of terms
    
    if rank == 0:
        print(f"Starting parallel computation (Block Distribution)")
        print(f"Number of processes: {size}")
        print(f"Number of terms: {N}")
        print("-" * 50)
    
    # Synchronize all processes before starting
    comm.Barrier()
    total_start_time = time.time()
    
    # Compute pi estimate
    pi_value, all_times = estimate_pi_parallel_block(N, comm)
    
    comm.Barrier()
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Print results from rank 0
    if rank == 0:
        print(f"\nEstimated π value: {pi_value:.10f}")
        print(f"Actual π value:    {3.141592653589793:.10f}")
        print(f"Error:             {abs(pi_value - 3.141592653589793):.10f}")
        print(f"\nTiming Information:")
        print(f"Total execution time: {total_time:.4f} seconds")
        print(f"\nPer-process execution times:")
        for i, t in enumerate(all_times):
            print(f"  Process {i}: {t:.4f} seconds")
        print("-" * 50)

if __name__ == "__main__":
    main()