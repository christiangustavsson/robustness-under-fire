import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import math
import matplotlib.pyplot as plt
import seaborn as sns


# Load the training and test datasets (from .npy files)
training_data = np.load('evaluation-1/datasets/target-dataset/training_data.npy')
test_data = np.load('evaluation-1/datasets/target-dataset/test_data.npy')

# Extract X and y values for training and testing
X_train = training_data[0, :].reshape(-1, 1)
y_train = training_data[1, :]
X_test = test_data[0, :].reshape(-1, 1)
y_test = test_data[1, :]

# Define the base kernel
base_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))

# Initialize the Gaussian Process Regressor
gpr = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=10)

# Define the hyperparameter grid
param_grid = {
    "kernel": [
        C(0.1, (1e-3, 1e3)) * RBF(0.1, (1e-3, 1e3)),
        C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)),
        C(10.0, (1e-3, 1e3)) * RBF(10.0, (1e-3, 1e3))
    ]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(gpr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the model with grid search
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Compute RMSE
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print(f"Best parameters from grid search: {grid_search.best_params_}")
print(f"RMSE on the test dataset: {rmse}")

# Plotting results
sns.set(style="white")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='orange', label='Predictions', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--', label='Ideal Line')
plt.title('Gaussian Process Regression: Prediction vs Ground Truth')
plt.xlabel('Ground Truth (y)')
plt.ylabel('Predicted Values (ŷ)')
plt.legend()
plt.show()
