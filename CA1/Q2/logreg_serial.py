# logreg_serial.py
import numpy as np
import time

class LogisticRegression:
    """
    Logistic Regression classifier with gradient descent (without sklearn)
    """
    def __init__(self, learning_rate=0.01, n_epochs=30):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
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
            
            # Print loss every 5 epochs
            if (epoch + 1) % 5 == 0:
                loss = -np.mean(y * np.log(y_predicted + 1e-15) + 
                               (1 - y) * np.log(1 - y_predicted + 1e-15))
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Binary predictions (n_samples,)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
    
    def accuracy(self, X, y):
        """
        Calculate accuracy
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

def load_and_merge_data():
    """
    Load all three datasets and merge them
    
    Data files:
    - data1.npy, data2.npy, data3.npy: feature matrices
    - labels1.npy, labels2.npy, labels3.npy: label vectors
    
    Returns:
        X: Combined features
        y: Combined labels (converted to binary {0, 1})
    """
    print("Loading data files...")
    
    # Load features
    X1 = np.load('data1.npy', allow_pickle=True)
    X2 = np.load('data2.npy', allow_pickle=True)
    X3 = np.load('data3.npy', allow_pickle=True)
    
    # Load labels
    y1 = np.load('labels1.npy', allow_pickle=True)
    y2 = np.load('labels2.npy', allow_pickle=True)
    y3 = np.load('labels3.npy', allow_pickle=True)
    
    print(f"Data1 shape: {X1.shape}, Labels1 shape: {y1.shape}")
    print(f"Data2 shape: {X2.shape}, Labels2 shape: {y2.shape}")
    print(f"Data3 shape: {X3.shape}, Labels3 shape: {y3.shape}")
    
    # Merge features and labels
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    # Check unique labels
    unique_labels = np.unique(y)
    print(f"\nUnique labels before conversion: {unique_labels}")
    
    # Convert labels to binary {0, 1}
    # Assuming labels are {-1, 1} or similar
    if len(unique_labels) == 2:
        # Map the smaller value to 0 and larger to 1
        min_label = unique_labels.min()
        max_label = unique_labels.max()
        y = ((y - min_label) / (max_label - min_label)).astype(int)
        print(f"Labels converted: {min_label} -> 0, {max_label} -> 1")
    else:
        print(f"Warning: Found {len(unique_labels)} unique labels: {unique_labels}")
        print("Attempting to binarize by threshold at 0")
        y = (y > 0).astype(int)
    
    return X, y

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets (without sklearn)
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def main():
    print("=" * 60)
    print("Serial Logistic Regression - Baseline")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and merge all datasets
    print("\nLoading and merging datasets...")
    try:
        X, y = load_and_merge_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nTotal samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Train the model
    print("\n" + "-" * 60)
    print("Training Logistic Regression Model")
    print("-" * 60)
    
    model = LogisticRegression(learning_rate=0.01, n_epochs=30)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Calculate accuracy
    train_accuracy = model.accuracy(X_train, y_train)
    test_accuracy = model.accuracy(X_test, y_test)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Train accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()