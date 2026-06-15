#!/usr/bin/env python3
# logreg_fedavg.py
"""
Federated Learning with Logistic Regression using MPI
Implements FedAvg algorithm for distributed training
"""

import numpy as np
import time
from mpi4py import MPI
import sys

class LogisticRegression:
    """
    Logistic Regression classifier with gradient descent
    """
    def __init__(self, learning_rate=0.01, n_epochs=1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def set_weights(self, weights, bias):
        """Set model weights from server"""
        self.weights = weights.copy()
        self.bias = bias
    
    def get_weights(self):
        """Get current model weights"""
        return self.weights.copy(), self.bias
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights if not set
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0
        
        # Gradient descent for n_epochs
        for epoch in range(self.n_epochs):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

def load_client_data(client_id):
    """
    Load data for a specific client
    
    Args:
        client_id: Client ID (1, 2, or 3)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load features and labels separately
    X = np.load(f'data{client_id}.npy', allow_pickle=True)
    y = np.load(f'labels{client_id}.npy', allow_pickle=True)
    
    # Flatten y if it's not 1D
    if y.ndim > 1:
        y = y.flatten()
    
    # Convert labels to binary {0, 1}
    unique_labels = np.unique(y)
    
    # Check if labels contain negative values
    if np.min(y) < 0:
        # Map to {0, 1}
        if len(unique_labels) == 2:
            min_label = unique_labels.min()
            max_label = unique_labels.max()
            y = ((y - min_label) / (max_label - min_label)).astype(int)
        else:
            y = (y > 0).astype(int)
    
    # Ensure y is integer type
    y = y.astype(int)
    
    # Split into train (80%) and test (20%)
    np.random.seed(42)
    n_samples = X.shape[0]
    n_test = int(n_samples * 0.2)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def server_process(comm, n_clients, n_rounds, n_features, local_epochs):
    """
    Server process for Federated Learning (FedAvg)
    
    Args:
        comm: MPI communicator
        n_clients: Number of clients
        n_rounds: Number of communication rounds
        n_features: Number of features
        local_epochs: Number of local epochs per round
    """
    print("=" * 70)
    print("SERVER: Federated Learning Started (FedAvg)")
    print("=" * 70)
    print(f"Number of clients: {n_clients}")
    print(f"Communication rounds: {n_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Number of features: {n_features}")
    print("=" * 70)
    
    # Initialize global model with zeros
    global_weights = np.zeros(n_features)
    global_bias = 0.0
    
    total_training_time = 0
    
    # FedAvg algorithm: iterate over communication rounds
    for round_num in range(n_rounds):
        print(f"\n{'='*70}")
        print(f"Communication Round {round_num + 1}/{n_rounds}")
        print('='*70)
        
        round_start_time = time.time()
        
        # Step 1: Broadcast global model to all clients
        for client_rank in range(1, n_clients + 1):
            comm.send((global_weights, global_bias), dest=client_rank, tag=round_num)
        
        print(f"SERVER: Sent global model to {n_clients} clients")
        
        # Step 2: Wait for all clients to receive and train
        # Clients will send back their updated models
        client_weights = []
        client_biases = []
        
        for client_rank in range(1, n_clients + 1):
            weights, bias = comm.recv(source=client_rank, tag=round_num)
            client_weights.append(weights)
            client_biases.append(bias)
        
        print(f"SERVER: Received updated models from {n_clients} clients")
        
        # Step 3: Aggregate models using FedAvg (simple averaging)
        global_weights = np.mean(client_weights, axis=0)
        global_bias = np.mean(client_biases)
        
        print(f"SERVER: Aggregated models using FedAvg")
        
        # Barrier: Synchronize all processes before next round
        comm.Barrier()
        
        round_time = time.time() - round_start_time
        total_training_time += round_time
        
        print(f"SERVER: Round {round_num + 1} completed in {round_time:.4f}s")
    
    print(f"\n{'='*70}")
    print("SERVER: Training completed")
    print(f"Total training time: {total_training_time:.4f} seconds")
    print('='*70)
    
    # Step 4: Send final global model to all clients for evaluation
    for client_rank in range(1, n_clients + 1):
        comm.send((global_weights, global_bias), dest=client_rank, tag=n_rounds)
    
    print("\nSERVER: Sent final model to all clients for evaluation")
    
    # Step 5: Receive accuracy from all clients
    client_accuracies = []
    for client_rank in range(1, n_clients + 1):
        accuracy = comm.recv(source=client_rank, tag=n_rounds + 1)
        client_accuracies.append(accuracy)
        print(f"SERVER: Client {client_rank} test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    avg_accuracy = np.mean(client_accuracies)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    print(f"Average test accuracy across all clients: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Total training time: {total_training_time:.4f} seconds")
    print('='*70)

def client_process(comm, rank, n_rounds, local_epochs, learning_rate):
    """
    Client process for Federated Learning
    
    Args:
        comm: MPI communicator
        rank: Client rank (1, 2, or 3)
        n_rounds: Number of communication rounds
        local_epochs: Number of local training epochs per round
        learning_rate: Learning rate for local training
    """
    client_id = rank  # Client ID is same as rank
    
    print(f"\nCLIENT {client_id}: Initializing...")
    
    try:
        # Load client's local data
        X_train, X_test, y_train, y_test = load_client_data(client_id)
    except Exception as e:
        print(f"CLIENT {client_id}: Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"CLIENT {client_id}: Data loaded successfully")
    print(f"CLIENT {client_id}: Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"CLIENT {client_id}: Features: {X_train.shape[1]}")
    
    # Check for valid labels before bincount
    if np.min(y_train) >= 0:
        print(f"CLIENT {client_id}: Train class distribution: {np.bincount(y_train)}")
    
    # Create local model
    model = LogisticRegression(learning_rate=learning_rate, n_epochs=local_epochs)
    
    total_training_time = 0
    
    # FedAvg algorithm: iterate over communication rounds
    for round_num in range(n_rounds):
        # Step 1: Receive global model from server
        global_weights, global_bias = comm.recv(source=0, tag=round_num)
        
        # Step 2: Set local model to global model
        model.set_weights(global_weights, global_bias)
        
        # Step 3: Train locally for local_epochs
        local_start_time = time.time()
        model.fit(X_train, y_train)
        local_training_time = time.time() - local_start_time
        total_training_time += local_training_time
        
        print(f"CLIENT {client_id}: Round {round_num + 1}/{n_rounds} - "
              f"Trained for {local_epochs} epoch(s) in {local_training_time:.4f}s")
        
        # Step 4: Send updated model to server
        weights, bias = model.get_weights()
        comm.send((weights, bias), dest=0, tag=round_num)
        
        # Barrier: Wait for all clients and server to synchronize
        comm.Barrier()
    
    print(f"CLIENT {client_id}: Total local training time: {total_training_time:.4f} seconds")
    
    # Step 5: Receive final global model
    final_weights, final_bias = comm.recv(source=0, tag=n_rounds)
    model.set_weights(final_weights, final_bias)
    
    # Step 6: Evaluate on local test set
    test_accuracy = model.accuracy(X_test, y_test)
    
    print(f"CLIENT {client_id}: Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Step 7: Send accuracy to server
    comm.send(test_accuracy, dest=0, tag=n_rounds + 1)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Configuration
    n_clients = size - 1  # All processes except rank 0
    
    # Get parameters from command line
    if len(sys.argv) >= 3:
        n_rounds = int(sys.argv[1])
        local_epochs = int(sys.argv[2])
    else:
        if rank == 0:
            print("ERROR: Missing arguments!")
            print("Usage: mpirun -n <num_processes> python logreg_fedavg.py <n_rounds> <local_epochs>")
            print("Example: mpirun -n 4 python logreg_fedavg.py 1 10")
        sys.exit(1)
    
    learning_rate = 0.01
    
    # Synchronize all processes before starting
    comm.Barrier()
    
    if rank == 0:
        # Server process
        # Get number of features from first client's data
        data = np.load('data1.npy', allow_pickle=True)
        n_features = data.shape[1]
        
        start_time = time.time()
        server_process(comm, n_clients, n_rounds, n_features, local_epochs)
        total_time = time.time() - start_time
        
        print(f"\nTotal execution time: {total_time:.4f} seconds")
    else:
        # Client process
        client_process(comm, rank, n_rounds, local_epochs, learning_rate)

if __name__ == "__main__":
    main()