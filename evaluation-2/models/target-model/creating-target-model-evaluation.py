import os, sys, datetime, pickle, shutil, random, json
import torchvision
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import Subset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Classes
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)

# Define data transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Fixed normalization typo
])

# Initialize global variables for best model tracking
global_best_accuracy = 0
global_best_model_wts = None

# Hyperparameter optimization function
def optimize_model(trial):
    global global_best_accuracy, global_best_model_wts

    # Hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

    # Model and data loaders
    model = resnet18(weights=None).to(device)  # Not using a pretrained model
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Load datasets
    path_training_dataset = "evaluation-2/datasets/training-dataset"
    path_test_dataset = "evaluation-2/datasets/test-dataset"

    training_dataset = datasets.ImageFolder(path_training_dataset, transform=transform)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.ImageFolder(path_test_dataset, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    n_epochs = 50
    best_accuracy = 0
    best_model_wts = None
    trial_logs = []

    for epoch in range(n_epochs):
        model.train()
        for j, (images, labels) in enumerate(training_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (j + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{j+1}/{len(training_dataloader)}], Loss: {loss.item():.4f}')
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc:.2f} %')

            # Log epoch details
            trial_logs.append({
                'epoch': epoch + 1,
                'accuracy': acc,
                'hyperparameters': trial.params
            })

            # Track the best model for this trial
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_wts = model.state_dict()

            # Update the global best model if this trial's model is better
            if best_accuracy > global_best_accuracy:
                global_best_accuracy = best_accuracy
                global_best_model_wts = best_model_wts

    # Attach logs to the trial for later access
    trial.set_user_attr("trial_logs", trial_logs)

    # Return the best accuracy for this trial
    return best_accuracy

# Optimize hyperparameters using Optuna
def main():
    global global_best_model_wts, global_best_accuracy

    # Create an Optuna study to maximize accuracy
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_model, n_trials=50)

    # Save the best model's weights across all trials
    if global_best_model_wts:
        torch.save(global_best_model_wts, "best_model.pth")

    # Save detailed logs of all trials
    all_logs = [trial.user_attrs['trial_logs'] for trial in study.trials]
    with open("trial_logs.json", "w") as f:
        json.dump(all_logs, f, indent=4)

    print(f"Best accuracy across all trials: {global_best_accuracy:.2f}")

if __name__ == "__main__":
    main()
