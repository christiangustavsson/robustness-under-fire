import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys

# Determine the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths and settings
subsets_folder = './evaluation-2/datasets/subsets'
test_dataset_folder = './evaluation-2/datasets/test-dataset'
log_file = "hyperparameter_optimization.log"
num_subsets = 20
models_to_evaluate = ["resnet18", "resnet152", "vgg11", "vgg19", "densenet121", "densenet161"]
num_trials = 20  # Optuna trials per model

# Redirect terminal output to log file
sys.stdout = open(log_file, "w")

# Initialize results storage
results_per_subset = {}

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load test dataset
test_dataset = datasets.ImageFolder(root=test_dataset_folder, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to define and optimize a model-specific objective
def optimize_model_for_subset(model_name, train_loader):
    def model_specific_objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'sgd'])

        # Adjust train_loader with new batch size
        subset_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)

        # Load the model
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif model_name == "resnet152":
            model = models.resnet152(weights=None)
        elif model_name == "vgg11":
            model = models.vgg11(weights=None)
        elif model_name == "vgg19":
            model = models.vgg19(weights=None)
        elif model_name == "densenet121":
            model = models.densenet121(weights=None)
        elif model_name == "densenet161":
            model = models.densenet161(weights=None)

        # Adjust the output layer for CIFAR-10
        if model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif model_name.startswith("vgg"):
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        elif model_name.startswith("densenet"):
            model.classifier = nn.Linear(model.classifier.in_features, 10)

        model.to(device)

        # Define optimizer and criterion
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(50):  # Fewer epochs for faster evaluation
            model.train()
            for inputs, labels in subset_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    # Perform the optimization for the current model
    study = optuna.create_study(direction='maximize')
    study.optimize(model_specific_objective, n_trials=num_trials)

    return study.best_params, study.best_value

# Iterate through subsets and models
for subset_idx in tqdm(range(1, num_subsets + 1), desc="Evaluating subsets"):
    subset_path = os.path.join(subsets_folder, f"subset_{subset_idx}")
    print(f"\nEvaluating models on {subset_path}")

    # Load subset dataset
    train_dataset = datasets.ImageFolder(root=subset_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Store results for this subset
    results_per_subset[subset_idx] = {}

    for model_name in tqdm(models_to_evaluate, desc=f"Subset {subset_idx} Models", leave=False):
        print(f"\nStarting optimization for {model_name} on subset_{subset_idx}")
        best_params, best_accuracy = optimize_model_for_subset(model_name, train_loader)
        results_per_subset[subset_idx][model_name] = {"best_params": best_params, "best_accuracy": best_accuracy}

        print(f"Subset {subset_idx}, Model {model_name}: Best Accuracy = {best_accuracy:.4f}")

# Save all results to a JSON file
with open("results_per_subset.json", "w") as f:
    json.dump(results_per_subset, f, indent=4)

# Restore terminal output
sys.stdout.close()
sys.stdout = sys.__stdout__

# Plot results
plt.figure(figsize=(12, 8))

# Plot each model's accuracy across subsets
for model_name in models_to_evaluate:
    accuracies = [results_per_subset[subset_idx][model_name]["best_accuracy"] for subset_idx in range(1, num_subsets + 1)]
    plt.plot(range(1, num_subsets + 1), accuracies, label=model_name)

# Set x-axis range from 1000 to 20000
plt.xlim(1000, 20000)

# Label the axes and title the plot
plt.xlabel('Subset Number', fontsize=14)
plt.ylabel('Best Accuracy', fontsize=14)
plt.title('Model Performance Across Subsets', fontsize=16)

# Show the legend and grid
plt.legend()
plt.grid()

# Tight layout for better spacing
plt.tight_layout()

# Save and display the plot
plt.savefig("performance_plot.png")
plt.show()
