import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# CNN Model


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Elastic Weight Consolidation (EWC)


class EWC:
    def __init__(self, model, dataloader, importance=5000):
        self.model = model
        self.dataloader = dataloader
        self.importance = importance
        self.params = {n: p for n, p in model.named_parameters()
                       if p.requires_grad}
        self.means = {}
        self.fisher_matrix = {}
        self._compute_fisher_matrix()

    def _compute_fisher_matrix(self):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}

        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            for n, p in self.params.items():
                fisher[n] += p.grad ** 2 / len(self.dataloader)

        self.fisher_matrix = fisher
        self.means = {n: p.clone() for n, p in self.params.items()}

    def penalty(self, model):
        penalty = 0
        for n, p in model.named_parameters():
            if n in self.fisher_matrix:
                fisher = self.fisher_matrix[n]
                mean = self.means[n]
                penalty += (fisher * (p - mean) ** 2).sum()
        return self.importance * penalty

# Training function with replay


def train_with_replay(model, dataloader, optimizer, criterion, ewc=None, replay_buffer=None):
    model.train()
    total_loss = 0

    # Combine replay buffer with current data
    if replay_buffer:
        combined_dataset = replay_buffer + dataloader.dataset
        combined_dataloader = DataLoader(
            combined_dataset, batch_size=64, shuffle=True)
    else:
        combined_dataloader = dataloader

    for inputs, labels in combined_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if ewc:
            loss += ewc.penalty(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(combined_dataloader)

# Testing function


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)

# Divide into class incremental tasks
tasks = [list(range(i, i + 2))
         for i in range(0, 10, 2)]  # [0-1], [2-3], ..., [8-9]

# Initialize model, criterion, and replay buffer
model = CNN(num_classes=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      weight_decay=1e-4)  # Use weight decay
criterion = nn.CrossEntropyLoss()
replay_buffer = []

# Incremental Learning Loop
ewc = None
buffer_size = 200  # Limit replay buffer size

for task_idx, task_classes in enumerate(tasks):
    print(f"Task {task_idx + 1}: Classes {task_classes}")

    # Create dataloaders for the current task
    task_train_dataset = [d for d in train_dataset if d[1] in task_classes]
    task_test_dataset = [d for d in test_dataset if d[1] in task_classes]

    task_train_loader = DataLoader(
        task_train_dataset, batch_size=64, shuffle=True)
    task_test_loader = DataLoader(
        task_test_dataset, batch_size=64, shuffle=False)

    # Train on the current task
    for epoch in range(2):  # Train for up to 10 epochs
        train_loss = train_with_replay(
            model, task_train_loader, optimizer, criterion, ewc, replay_buffer)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        # Check validation accuracy on the current task
        val_accuracy = test(model, task_test_loader)
        print(
            f"Validation Accuracy on Task {task_idx + 1}: {val_accuracy:.2f}%")
        if val_accuracy > 98:  # Early stopping if accuracy is high
            break

    # Evaluate on all tasks seen so far
    for past_task_idx in range(task_idx + 1):
        past_task_classes = tasks[past_task_idx]
        past_task_test_dataset = [
            d for d in test_dataset if d[1] in past_task_classes]
        past_task_test_loader = DataLoader(
            past_task_test_dataset, batch_size=64, shuffle=False)
        accuracy = test(model, past_task_test_loader)
        print(
            f"Accuracy on Task {past_task_idx + 1} ({past_task_classes}): {accuracy:.2f}%")

    # Update EWC and replay buffer
    ewc = EWC(model, task_train_loader, importance=5000)
    replay_buffer.extend(
        [(x, y) for x, y in task_train_dataset][:buffer_size // len(tasks)])
