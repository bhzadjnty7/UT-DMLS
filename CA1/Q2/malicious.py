#!/usr/bin/env python3
# malicious.py
"""
Federated Learning with Malicious Client
Client 2 uses learning rate 0.5 instead of 0.01
"""

import numpy as np
import time
from mpi4py import MPI

class LogisticRegression:
    """Logistic Regression classifier with gradient descent"""
    def __init__(self, learning_rate=0.01, n_epochs=1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def set_weights(self, weights, bias):
        self.weights = weights.copy()
        self.bias = bias
    
    def get_weights(self):
        return self.weights.copy(), self.bias
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0
        
        for epoch in range(self.n_epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def load_client_data(client_id):
    X = np.load(f'data{client_id}.npy', allow_pickle=True)
    y = np.load(f'labels{client_id}.npy', allow_pickle=True)
    
    if y.ndim > 1:
        y = y.flatten()
    
    unique_labels = np.unique(y)
    if np.min(y) < 0:
        if len(unique_labels) == 2:
            min_label = unique_labels.min()
            max_label = unique_labels.max()
            y = ((y - min_label) / (max_label - min_label)).astype(int)
        else:
            y = (y > 0).astype(int)
    
    y = y.astype(int)
    
    np.random.seed(42)
    n_samples = X.shape[0]
    n_test = int(n_samples * 0.2)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def server_process(comm, n_clients, n_rounds, n_features, local_epochs):
    print("=" * 70)
    print("SERVER: Federated Learning with Malicious Client")
    print("=" * 70)
    print(f"Number of clients: {n_clients}")
    print(f"Communication rounds: {n_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"⚠️  WARNING: Client 2 has malicious learning rate (0.5)")
    print("=" * 70)
    
    global_weights = np.zeros(n_features)
    global_bias = 0.0
    total_training_time = 0
    
    for round_num in range(n_rounds):
        print(f"\n{'='*70}")
        print(f"Communication Round {round_num + 1}/{n_rounds}")
        print('='*70)
        
        round_start_time = time.time()
        
        for client_rank in range(1, n_clients + 1):
            comm.send((global_weights, global_bias), dest=client_rank, tag=round_num)
        
        print(f"SERVER: Sent global model to {n_clients} clients")
        
        client_weights = []
        client_biases = []
        
        for client_rank in range(1, n_clients + 1):
            weights, bias = comm.recv(source=client_rank, tag=round_num)
            client_weights.append(weights)
            client_biases.append(bias)
        
        print(f"SERVER: Received updated models from {n_clients} clients")
        
        global_weights = np.mean(client_weights, axis=0)
        global_bias = np.mean(client_biases)
        
        print(f"SERVER: Aggregated models using FedAvg")
        
        comm.Barrier()
        
        round_time = time.time() - round_start_time
        total_training_time += round_time
        
        print(f"SERVER: Round {round_num + 1} completed in {round_time:.4f}s")
    
    print(f"\n{'='*70}")
    print("SERVER: Training completed")
    print(f"Total training time: {total_training_time:.4f} seconds")
    print('='*70)
    
    for client_rank in range(1, n_clients + 1):
        comm.send((global_weights, global_bias), dest=client_rank, tag=n_rounds)
    
    print("\nSERVER: Sent final model to all clients for evaluation")
    
    client_accuracies = []
    for client_rank in range(1, n_clients + 1):
        accuracy = comm.recv(source=client_rank, tag=n_rounds + 1)
        client_accuracies.append(accuracy)
        marker = "⚠️ " if client_rank == 2 else ""
        print(f"SERVER: {marker}Client {client_rank} test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    avg_accuracy = np.mean(client_accuracies)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS (WITH MALICIOUS CLIENT)")
    print('='*70)
    print(f"Average test accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Total training time: {total_training_time:.4f} seconds")
    print('='*70)

def client_process(comm, rank, n_rounds, local_epochs, learning_rate, is_malicious=False):
    client_id = rank
    
    if is_malicious:
        learning_rate = 0.5
        print(f"\n⚠️  CLIENT {client_id}: MALICIOUS - Using learning rate {learning_rate}")
    
    print(f"\nCLIENT {client_id}: Initializing...")
    
    X_train, X_test, y_train, y_test = load_client_data(client_id)
    
    print(f"CLIENT {client_id}: Data loaded successfully")
    print(f"CLIENT {client_id}: Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"CLIENT {client_id}: Learning rate: {learning_rate}")
    
    if np.min(y_train) >= 0:
        print(f"CLIENT {client_id}: Train class distribution: {np.bincount(y_train)}")
    
    model = LogisticRegression(learning_rate=learning_rate, n_epochs=local_epochs)
    total_training_time = 0
    
    for round_num in range(n_rounds):
        global_weights, global_bias = comm.recv(source=0, tag=round_num)
        model.set_weights(global_weights, global_bias)
        
        local_start_time = time.time()
        model.fit(X_train, y_train)
        local_training_time = time.time() - local_start_time
        total_training_time += local_training_time
        
        marker = "⚠️ " if is_malicious else ""
        print(f"{marker}CLIENT {client_id}: Round {round_num + 1}/{n_rounds} - "
              f"Trained for {local_epochs} epoch(s) in {local_training_time:.4f}s")
        
        weights, bias = model.get_weights()
        comm.send((weights, bias), dest=0, tag=round_num)
        
        comm.Barrier()
    
    print(f"CLIENT {client_id}: Total local training time: {total_training_time:.4f} seconds")
    
    final_weights, final_bias = comm.recv(source=0, tag=n_rounds)
    model.set_weights(final_weights, final_bias)
    
    test_accuracy = model.accuracy(X_test, y_test)
    
    marker = "⚠️ " if is_malicious else ""
    print(f"{marker}CLIENT {client_id}: Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    comm.send(test_accuracy, dest=0, tag=n_rounds + 1)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n_clients = size - 1
    n_rounds = 10
    local_epochs = 1
    learning_rate = 0.01
    
    comm.Barrier()
    
    if rank == 0:
        data = np.load('data1.npy', allow_pickle=True)
        n_features = data.shape[1]
        
        start_time = time.time()
        server_process(comm, n_clients, n_rounds, n_features, local_epochs)
        total_time = time.time() - start_time
        
        print(f"\nTotal execution time: {total_time:.4f} seconds")
    else:
        is_malicious = (rank == 2)
        client_process(comm, rank, n_rounds, local_epochs, learning_rate, is_malicious)

if __name__ == "__main__":
    main()