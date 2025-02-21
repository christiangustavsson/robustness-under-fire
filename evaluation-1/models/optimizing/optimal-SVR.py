from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and test datasets (from .npy files)
training_data = np.load('evaluation-1/datasets/target-dataset/training_data.npy')
test_data = np.load('evaluation-1/datasets/target-dataset/test_data.npy')

# The first row of each dataset is the x-values, and the second row is the y-values
X_train = training_data[0, :].reshape(-1, 1)  # X values (features) for training
y_train = training_data[1, :]  # y values (targets) for training

X_test = test_data[0, :].reshape(-1, 1)  # X values (features) for testing
y_test = test_data[1, :]  # y values (targets) for testing

# Initialize the SVR model with the RBF kernel
svr_model = SVR(kernel='rbf')

# Define the hyperparameter grid, including epsilon
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],       # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1, 10],  # RBF kernel coefficient
    'epsilon': [0.001, 0.01, 0.1, 1]      # Epsilon in the loss function
}

# Initialize GridSearchCV to search for the best parameters
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Calculate the RMSE (Root Mean Squared Error)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# Output the results
print(f"Best parameters from grid search: {grid_search.best_params_}")
print(f"RMSE on the test dataset: {rmse}")

# Plotting the results: True values vs. Predicted values

# Set Seaborn white theme
sns.set(style="white")

plt.figure(figsize=(10, 6))

# Plot the true values vs predicted values
plt.scatter(y_test, y_pred, color='purple', label='Predictions', alpha=0.7)

# Plot the ideal line where true = predicted
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')

# Adding labels and title
plt.title('SVR with RBF Kernel: Prediction vs Ground Truth')
plt.xlabel(r'Ground Truth (y)')
plt.ylabel(r'Prediction Values ($\hat{y}$)')
plt.legend()

# Show the plot
plt.show()
