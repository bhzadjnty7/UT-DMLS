#!/usr/bin/env python3
"""
Distributed Training with Gradient Accumulation
Course: Distributed Machine Learning Systems (Fall 1404)
Assignment 4 - Question 1

This script implements distributed training using PyTorch DDP
with gradient accumulation for memory management.
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    """Training configuration parameters."""
    # Data paths - Updated for actual data structure
    DATA_DIR = Path(os.environ.get('DATA_DIR', os.path.expanduser('~/data')))
    TRAIN_X_PATH = DATA_DIR / 'train_data' / 'train_x.npy'
    TRAIN_Y_PATH = DATA_DIR / 'train_data' / 'train_y.npy'
    TEST_X_PATH = DATA_DIR / 'test_data' / 'test_x.npy'
    TEST_Y_PATH = DATA_DIR / 'test_data' / 'test_y.npy'
    
    # Model checkpoint path
    CHECKPOINT_DIR = Path(os.environ.get('CHECKPOINT_DIR', './checkpoints'))
    
    # Training hyperparameters
    PHYSICAL_BATCH_SIZE = 32      # Actual batch size per forward pass
    ACCUMULATION_STEPS = 4         # Number of accumulation steps
    EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * ACCUMULATION_STEPS  # = 128
    
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    
    # Model architecture
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATE = 0.2


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
        # Load and convert features from float16 to float32
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        
        print(f"[Dataset] Loaded features shape: {self.features.shape}")
        print(f"[Dataset] Loaded labels shape: {self.labels.shape}")
        print(f"[Dataset] Number of classes: {len(np.unique(self.labels))}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ==============================================================================
# Model Architecture
# ==============================================================================
class FeedForwardClassifier(nn.Module):
    """
    Feed-forward neural network classifier with BatchNorm and ReLU activations.
    Designed to achieve >80% accuracy on the test set.
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
        
        # Initialize weights
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
        """Forward pass through the network."""
        return self.network(x)


# ==============================================================================
# Distributed Training Utilities
# ==============================================================================
def setup_distributed():
    """
    Initialize distributed training environment.
    Uses 'gloo' backend for CPU-only clusters (Raspberry Pi).
    
    Returns:
        tuple: (rank, world_size, local_rank)
    """
    # Initialize process group
    dist.init_process_group(backend='gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"[Rank {rank}] Initialized: world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                    epoch: int, accuracy: float, path: Path):
    """
    Save model checkpoint (only from rank 0).
    
    Args:
        model: The model to save (DDP wrapped)
        optimizer: The optimizer state
        epoch: Current epoch number
        accuracy: Current test accuracy
        path: Path to save checkpoint
    """
    # Extract the underlying model from DDP wrapper
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[Rank 0] Checkpoint saved to {path}")


# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch_with_accumulation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    rank: int,
    accumulation_steps: int = 4,
    use_gradient_accumulation: bool = True
):
    """
    Train for one epoch with gradient accumulation.
    
    Key insight for gradient accumulation with DDP:
    - DDP automatically synchronizes gradients in backward() via all-reduce
    - For intermediate accumulation steps, we DON'T want synchronization
    - Use model.no_sync() context manager to skip sync in intermediate steps
    - Only sync gradients on the final accumulation step
    
    Args:
        model: DDP-wrapped model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        rank: Process rank
        accumulation_steps: Number of steps to accumulate gradients
        use_gradient_accumulation: Whether to use gradient accumulation
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Determine if this is an accumulation step or update step
        is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        
        if use_gradient_accumulation and is_accumulation_step and not is_last_batch:
            # Intermediate accumulation step: skip gradient synchronization
            # This reduces communication overhead significantly
            with model.no_sync():
                outputs = model(features)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps for correct gradient magnitude
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
        else:
            # Final accumulation step or last batch: allow gradient sync
            outputs = model(features)
            loss = criterion(outputs, labels)
            scaled_loss = loss / accumulation_steps if use_gradient_accumulation else loss
            scaled_loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics (use unscaled loss for reporting)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        criterion: Loss function
    
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
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed Training with Gradient Accumulation')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=Config.PHYSICAL_BATCH_SIZE,
                        help='Physical batch size per process')
    parser.add_argument('--accumulation-steps', type=int, default=Config.ACCUMULATION_STEPS,
                        help='Gradient accumulation steps')
    parser.add_argument('--no-accumulation', action='store_true',
                        help='Disable gradient accumulation')
    parser.add_argument('--checkpoint-name', type=str, default='model_checkpoint.pt',
                        help='Checkpoint filename')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Determine if using gradient accumulation
    use_accumulation = not args.no_accumulation
    effective_batch_size = args.batch_size * args.accumulation_steps if use_accumulation else args.batch_size
    
    if rank == 0:
        print("=" * 60)
        print("Distributed Training Configuration")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Physical batch size: {args.batch_size}")
        print(f"Accumulation steps: {args.accumulation_steps if use_accumulation else 'N/A'}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Gradient accumulation: {'Enabled' if use_accumulation else 'Disabled'}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print("=" * 60)
    
    # Load datasets
    try:
        train_dataset = NPYDataset(Config.TRAIN_X_PATH, Config.TRAIN_Y_PATH)
        test_dataset = NPYDataset(Config.TEST_X_PATH, Config.TEST_Y_PATH)
    except FileNotFoundError as e:
        print(f"[Rank {rank}] Error loading data: {e}")
        print(f"[Rank {rank}] Please ensure data files exist at:")
        print(f"  - {Config.TRAIN_X_PATH}")
        print(f"  - {Config.TRAIN_Y_PATH}")
        print(f"  - {Config.TEST_X_PATH}")
        print(f"  - {Config.TEST_Y_PATH}")
        cleanup_distributed()
        return
    
    # Get data dimensions
    input_dim = train_dataset.features.shape[1]
    num_classes = len(np.unique(train_dataset.labels))
    
    if rank == 0:
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Set to 0 for Raspberry Pi
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
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
    
    # Wrap model with DDP
    model = DDP(model, find_unused_parameters=False)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    best_accuracy = 0.0
    start_time = time.time()
    
    if rank == 0:
        print("\nStarting training...")
        print("-" * 60)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch_with_accumulation(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            rank=rank,
            accumulation_steps=args.accumulation_steps,
            use_gradient_accumulation=use_accumulation
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Aggregate metrics across all processes
        metrics = torch.tensor([train_loss, train_acc, test_loss, test_acc])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        
        train_loss, train_acc, test_loss, test_acc = metrics.tolist()
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                checkpoint_path = Config.CHECKPOINT_DIR / args.checkpoint_name
                save_checkpoint(model, optimizer, epoch, test_acc, checkpoint_path)
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print("-" * 60)
        print(f"Training completed!")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Best test accuracy: {best_accuracy:.2f}%")
        print("=" * 60)
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()