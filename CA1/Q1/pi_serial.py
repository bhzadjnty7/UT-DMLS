import time

def compute_term(k):
    """
    Compute a single term of the Euler series with computational cost proportional to k
    """
    s = 0
    for i in range(k // 1000):
        s += i * i
    return 1.0 / (k * k)

def estimate_pi_serial(N):
    """
    Serial implementation to estimate pi using Euler's series
    
    Args:
        N: Number of terms to compute
    
    Returns:
        Estimated value of pi
    """
    sum_terms = 0.0
    
    # Compute sum of series terms
    for k in range(1, N + 1):
        sum_terms += compute_term(k)
    
    # Estimate pi using the formula: pi ≈ sqrt(6 * sum)
    pi_estimate = (6.0 * sum_terms) ** 0.5
    
    return pi_estimate

def main():
    N = 500000  # Number of terms
    
    print(f"Starting serial computation with N = {N}")
    print("-" * 50)
    
    # Measure execution time
    start_time = time.time()
    
    pi_value = estimate_pi_serial(N)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print(f"Estimated π value: {pi_value:.10f}")
    print(f"Actual π value:    {3.141592653589793:.10f}")
    print(f"Error:             {abs(pi_value - 3.141592653589793):.10f}")
    print(f"Execution time:    {execution_time:.4f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    main()