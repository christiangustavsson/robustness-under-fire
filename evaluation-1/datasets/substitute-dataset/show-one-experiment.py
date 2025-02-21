import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved experiments from disk
all_experiments = np.load('evaluation-1/datasets/substitute-dataset/all_experiments.npy', allow_pickle=True)

# Specify the experiment you want to load (for example, experiment 4)
e = 4  # Experiment index (0-based indexing)

# Retrieve the e-th experiment
experiment_data = all_experiments[e]

# Print the experiment data (inputs and predictions for that experiment)
print(f"Experiment {e+1} data:")
print(experiment_data)

# Extract inputs and predictions from the experiment data
inputs = experiment_data[:, 0]  # First column: input values
predictions = experiment_data[:, 1]  # Second column: predicted values

# Set Seaborn theme to 'white'
sns.set_theme(style='white')

# Create a plot with Seaborn and Matplotlib
plt.figure(figsize=(10, 6))

# Scatter plot of inputs vs predictions
plt.scatter(inputs, predictions, color='blue', label=f'Experiment {e+1}', alpha=0.7)

# Adding labels and title
plt.title(f'Experiment {e+1} - Input vs Prediction', fontsize=16)
plt.xlabel('Input Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)

# Display grid
plt.grid(True)

# Show the legend
plt.legend()

# Show the plot
plt.show()
