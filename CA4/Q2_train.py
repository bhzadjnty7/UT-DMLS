#!/usr/bin/env python3
"""
Distributed Training with Hugging Face Accelerate
Course: Distributed Machine Learning Systems (Fall 1404)
Assignment 4 - Question 2

This script rewrites Q1 using HuggingFace Accelerate library
for simplified distributed training and gradient accumulation.
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hugging Face Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed


# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    """Training configuration parameters."""
    # Data paths
    DATA_DIR = Path(os.environ.get('DATA_DIR', os.path.expanduser('~/data')))
    TRAIN_X_PATH = DATA_DIR / 'train_data' / 'train_x.npy'
    TRAIN_Y_PATH = DATA_DIR / 'train_data' / 'train_y.npy'
    TEST_X_PATH = DATA_DIR / 'test_data' / 'test_x.npy'
    TEST_Y_PATH = DATA_DIR / 'test_data' / 'test_y.npy'
    
    # Checkpoint directory
    CHECKPOINT_DIR = Path(os.environ.get('CHECKPOINT_DIR', './checkpoints'))
    
    # Training hyperparameters
    PHYSICAL_BATCH_SIZE = 32
    ACCUMULATION_STEPS = 4
    EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS  # = 128
    
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    
    # Model architecture
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATE = 0.2
    
    # Random seed for reproducibility
    SEED = 42


# ==============================================================================
# Dataset
# ==============================================================================
class NPYDataset(Dataset):
    """
    Custom Dataset for loading .npy files.
    Converts float16 features to float32 for training compatibility.
    """
    
    def __init__(self, features_path: Path, labels_path: Path):
        """
        Initialize dataset by loading numpy arrays.
        
        Args:
            features_path: Path to features .npy file
            labels_path: Path to labels .npy file
        """
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ==============================================================================
# Model Architecture (Same as Q1)
# ==============================================================================
class FeedForwardClassifier(nn.Module):
    """
    Feed-forward neural network classifier with BatchNorm and ReLU activations.
    Same architecture as Q1 for fair comparison.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = Config.HIDDEN_DIMS
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with BatchNorm and ReLU
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ==============================================================================
# Training Functions with Accelerate
# ==============================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accelerator: Accelerator
):
    """
    Train for one epoch using Accelerate's gradient accumulation.
    
    Key difference from Q1:
    - No manual no_sync() management needed
    - accelerator.accumulate() handles everything automatically
    - Much cleaner and simpler code
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        accelerator: Accelerate instance
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Accelerate handles gradient accumulation automatically
        # No need for manual no_sync() or loss scaling
        with accelerator.accumulate(model):
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator
):
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        criterion: Loss function
        accelerator: Accelerate instance
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Gather predictions from all processes
            predictions = outputs.argmax(dim=1)
            
            # Gather all results for accurate metrics
            all_predictions = accelerator.gather(predictions)
            all_labels = accelerator.gather(labels)
            all_losses = accelerator.gather(loss.repeat(len(labels)))
            
            if accelerator.is_main_process:
                total_loss += all_losses.mean().item()
                total += all_labels.size(0)
                correct += (all_predictions == all_labels).sum().item()
    
    # Only main process has accurate metrics
    if accelerator.is_main_process:
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
    else:
        avg_loss = 0.0
        accuracy = 0.0
    
    return avg_loss, accuracy


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    """Main training function using Accelerate."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Distributed Training with HuggingFace Accelerate'
    )
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=Config.PHYSICAL_BATCH_SIZE,
                        help='Physical batch size per process')
    parser.add_argument('--accumulation-steps', type=int, default=Config.ACCUMULATION_STEPS,
                        help='Gradient accumulation steps')
    parser.add_argument('--checkpoint-name', type=str, default='model_checkpoint.pt',
                        help='Checkpoint filename')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(Config.SEED)
    
    # Initialize Accelerator with gradient accumulation
    # This is the KEY difference from Q1 - Accelerate handles everything!
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        cpu=True  # Force CPU for Raspberry Pi cluster
    )
    
    # Calculate effective batch size
    effective_batch_size = (
        args.batch_size * 
        args.accumulation_steps * 
        accelerator.num_processes
    )
    
    # Print configuration (only main process)
    if accelerator.is_main_process:
        print("=" * 60)
        print("Distributed Training with HuggingFace Accelerate")
        print("=" * 60)
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Physical batch size: {args.batch_size}")
        print(f"Accumulation steps: {args.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Device: {accelerator.device}")
        print("=" * 60)
    
    # Load datasets
    try:
        train_dataset = NPYDataset(Config.TRAIN_X_PATH, Config.TRAIN_Y_PATH)
        test_dataset = NPYDataset(Config.TEST_X_PATH, Config.TEST_Y_PATH)
    except FileNotFoundError as e:
        if accelerator.is_main_process:
            print(f"Error loading data: {e}")
        return
    
    # Get data dimensions
    input_dim = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))
    
    if accelerator.is_main_process:
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model
    model = FeedForwardClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, 
    )
    
    # Prepare everything with Accelerator
    # This is where the magic happens - Accelerate wraps everything
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    # Training loop
    best_accuracy = 0.0
    start_time = time.time()
    
    if accelerator.is_main_process:
        print("\nStarting training...")
        print("-" * 60)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            accelerator=accelerator
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            accelerator=accelerator
        )
        
        # Update learning rate
        scheduler.step(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress (only main process)
        if accelerator.is_main_process:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model (only main process via Accelerator)
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                checkpoint_path = Config.CHECKPOINT_DIR / args.checkpoint_name
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc
                }
                accelerator.save(checkpoint, checkpoint_path)
                print(f"  -> New best model saved! Accuracy: {test_acc:.2f}%")
    
    total_time = time.time() - start_time
    
    # Final summary
    if accelerator.is_main_process:
        print("-" * 60)
        print(f"Training completed!")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Best test accuracy: {best_accuracy:.2f}%")
        print("=" * 60)


if __name__ == '__main__':
    main()