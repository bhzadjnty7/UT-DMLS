# Classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

class ConvNet(nn.Module):
    """
    Convolutional network for STL-10 classification
    Goal: Achieve >50% accuracy on test set
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)  # 96x96 -> 48x48
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)  # 48x48 -> 24x24
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 12x12 -> 6x6
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


def get_data_loaders(batch_size=32, data_path='/storage/dmls/stl10_data/'):
    """
    Prepare DataLoaders for STL-10
    """
    # Data Augmentation for Train
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], 
                           std=[0.2603, 0.2566, 0.2713])
    ])
    
    # Transform for Test (without augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4467, 0.4398, 0.4066], 
                           std=[0.2603, 0.2566, 0.2713])
    ])
    
    # Load datasets
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
    
    # Create DataLoaders with persistent_workers=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.3f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set
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
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


def main():
    # Settings
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    print(f'Batch size: {batch_size}')
    print(f'Number of epochs: {num_epochs}')
    print('-' * 60)
    
    # Load data
    print('Loading data...')
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    print('-' * 60)
    
    # Model, Loss, Optimizer
    model = ConvNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Track memory
    torch.cuda.reset_peak_memory_stats(device)
    
    # Training
    print('Starting training...')
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s')
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'New best accuracy: {best_acc:.2f}%')
    
    total_time = time.time() - start_time
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    
    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)')
    print(f'Best test accuracy: {best_acc:.2f}%')
    print(f'Peak GPU memory: {max_memory:.2f} GB')
    print('=' * 60)
    
    return {
        'best_accuracy': best_acc,
        'total_time': total_time,
        'peak_memory_gb': max_memory
    }


if __name__ == '__main__':
    results = main()