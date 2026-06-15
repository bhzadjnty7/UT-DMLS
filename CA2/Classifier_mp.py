# Classifier_mp.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import datasets, transforms
import time
import os
import socket


def find_free_port():
    """
    Find a free port for the master node
    This function is necessary to avoid conflicts when using concurrently
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ConvNet(nn.Module):
    """
    Same convolutional network as before
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def setup_ddp(rank, world_size, backend, master_port):
    """
    Setup DDP environment
    
    Args:
        rank: process ID (0 or 1)
        world_size: total number of processes (2)
        backend: communication backend type ('nccl' or 'gloo')
        master_port: master node port
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """
    Clean up and close process group
    """
    dist.destroy_process_group()


def get_data_loaders(rank, world_size, batch_size=32, 
                     data_path='/storage/dmls/stl10_data/'):
    """
    Prepare DataLoader with DistributedSampler
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], 
                           std=[0.2603, 0.2566, 0.2713])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], 
                           std=[0.2603, 0.2566, 0.2713])
    ])
    
    # Datasets
    train_dataset = datasets.STL10(
        root=data_path,
        split='train',
        transform=train_transform,
        download=False
    )
    
    test_dataset = datasets.STL10(
        root=data_path,
        split='test',
        transform=test_transform,
        download=False
    )
    
    # DistributedSampler to distribute data across GPUs
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
    
    # DataLoaders with persistent_workers=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, rank):
    """
    Train one epoch with DDP
    """
    model.train()
    train_loader.sampler.set_epoch(rank)  # For proper shuffling
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if rank == 0 and (batch_idx + 1) % 50 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    # Aggregate metrics from all processes
    metrics = torch.tensor([running_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    epoch_loss = metrics[0].item() / (len(train_loader) * dist.get_world_size())
    epoch_acc = 100. * metrics[1].item() / metrics[2].item()
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device, rank):
    """
    Evaluate model with DDP
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Aggregate metrics
    metrics = torch.tensor([test_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    test_loss = metrics[0].item() / (len(test_loader) * dist.get_world_size())
    test_acc = 100. * metrics[1].item() / metrics[2].item()
    
    return test_loss, test_acc


def train_ddp(rank, world_size, backend, master_port, batch_size, num_epochs):
    """
    Main training function for each process
    """
    # Setup DDP
    setup_ddp(rank, world_size, backend, master_port)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f'Using backend: {backend}')
        print(f'Using {world_size} GPUs')
        print(f'Batch size per GPU: {batch_size}')
        print(f'Total batch size: {batch_size * world_size}')
        print('-' * 60)
    
    # Load data
    train_loader, test_loader = get_data_loaders(
        rank, world_size, batch_size=batch_size
    )
    
    if rank == 0:
        print(f'Train samples: {len(train_loader.dataset)}')
        print(f'Test samples: {len(test_loader.dataset)}')
        print('-' * 60)
    
    # Model
    model = ConvNet(num_classes=10).to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Track memory
    torch.cuda.reset_peak_memory_stats(device)
    
    # Training
    if rank == 0:
        print('Starting training...')
        start_time = time.time()
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        if rank == 0:
            epoch_start = time.time()
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, rank
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, rank)
        scheduler.step()
        
        if rank == 0:
            epoch_time = time.time() - epoch_start
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'New best accuracy: {best_acc:.2f}%')
    
    if rank == 0:
        total_time = time.time() - start_time
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print('\n' + '=' * 60)
        print('Training completed!')
        print(f'Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)')
        print(f'Best test accuracy: {best_acc:.2f}%')
        print(f'Peak GPU memory (rank 0): {max_memory:.2f} GB')
        print('=' * 60)
    
    cleanup_ddp()


def main():
    """
    Main function to run DDP
    """
    # Settings
    world_size = 2  # Number of GPUs
    backend = 'nccl'  # Communication backend (nccl or gloo)
    batch_size = 32  # batch size per GPU
    num_epochs = 30
    
    # Find a free port
    master_port = find_free_port()
    print(f'Master port: {master_port}')
    
    # Launch multiprocessing
    mp.spawn(
        train_ddp,
        args=(world_size, backend, master_port, batch_size, num_epochs),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()