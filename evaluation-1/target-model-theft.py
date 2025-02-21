import numpy as np
import joblib  # or pickle if you're using pickle for model loading
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the saved model
model = joblib.load('evaluation-1/models/target-model/best_ridge_model.pkl')  # Load your best model

# Number of experiments
n_experiments = 10000

# Initialize a list to store all experiments
all_experiments = []

# Query the model n_experiments number of times with random float inputs
for experiment in tqdm(range(n_experiments), desc="Running Experiments", unit="experiment"):
    inputs_and_predictions = []  # Initialize the list for each experiment

    # Run the experiment (100 queries)
    for _ in range(100):
        # Generate a random input value (float) between 0 and 10
        X_input = np.random.uniform(0, 10, size=(1, 1))  # Random float between 0 and 10

        # Make prediction
        prediction = model.predict(X_input)

        # Store the input and prediction as a tuple
        inputs_and_predictions.append([X_input[0][0], prediction[0]])

    # Convert the list of inputs and predictions to a Numpy array
    inputs_and_predictions_array = np.array(inputs_and_predictions)

    # Add the sorted array to the list of all experiments
    all_experiments.append(inputs_and_predictions_array)

# Save all experiments to disk
# Uncomment the following line to save the experiments to disk
# np.save('evaluation-1/datasets/substitute-dataset/all_experiments.npy', all_experiments)

print(f"All {n_experiments} experiments completed and saved to disk.")
