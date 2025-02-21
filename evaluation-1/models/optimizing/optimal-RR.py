import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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

# Create a pipeline with PolynomialFeatures and Ridge regression
model_pipeline = make_pipeline(PolynomialFeatures(), Ridge())

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'ridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Polynomial degree
}

# Initialize GridSearchCV to search for the best parameters
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

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

# Increase font size by adjusting rcParams for various plot elements
plt.rcParams.update({
    'axes.labelsize': 18,    # Axis labels font size
    'axes.titlesize': 18,    # Title font size
    'xtick.labelsize': 18,   # X-axis tick labels font size
    'ytick.labelsize': 18,   # Y-axis tick labels font size
    'legend.fontsize': 18,   # Legend font size
    'figure.titlesize': 22,  # Figure title font size
})

plt.figure(figsize=(10, 6))

# Plot the true values vs predicted values
plt.scatter(y_test, y_pred, color='blue', label='Predictions', alpha=0.7)

# Plot the ideal line where true = predicted
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')

# Adding labels and title
plt.title('Ridge Regression: Prediction vs Ground Truth')
plt.xlabel(r'Ground Truth (y)')
plt.ylabel(r'Prediction Values ($\hat{y}$)')
plt.legend()

# Show the plot
plt.show()

# Save the best model using pickle. 
# NOTE! Only uncomment if a new models is to be saved!
# with open('evaluation-1/models/target-model/best_ridge_model.pkl', 'wb') as f:
#     pickle.dump(best_model, f)

# print("Model saved successfully with pickle!")
