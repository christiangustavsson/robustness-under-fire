import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Non-monotonic function using a sum of sine and cosine terms with a quadratic term
def non_monotonic_function(x):
    return 3 * np.sin(2 * x) - 2 * np.cos(3 * x) + 0.5 * x**2 - 4

if __name__ == '__main__':
    # Generate 200 points from 0 to 10
    x_values = np.linspace(0, 10, 1000)
    
    # Calculate y-values using the non-monotonic function
    y_values = non_monotonic_function(x_values)
    
    # Randomly sample 40 values for training and 10 values for test
    np.random.seed(42)  # Set random seed for reproducibility
    training_indices = np.random.choice(len(y_values), size=80, replace=False)  # Random indices for training
    test_indices = np.random.choice(len(y_values), size=20, replace=False)  # Random indices for test
    
    # Create the training and test arrays (x and y values)
    training_x = x_values[training_indices]  # x values for training
    training_y = y_values[training_indices]  # y values (function values) for training
    test_x = x_values[test_indices]          # x values for test
    test_y = y_values[test_indices]          # y values (function values) for test
    
    # Convert the training and test data into numpy arrays (if not already)
    training_data = np.array([training_x, training_y])
    test_data = np.array([test_x, test_y])
    
    # Save the training and test datasets (x and y values) to disk using pickle
    # Uncomment to run the code, if no training_data.npy and test_data.npy files exist
    # with open('evaluation-1/datasets/target-dataset/training_data.npy', 'wb') as f:
    #     np.save(f, training_data)  # Save the training data as a numpy array
    # with open('evaluation-1/datasets/target-dataset/test_data.npy2', 'wb') as f:
    #     np.save(f, test_data)  # Save the test data as a numpy array
    
    # Set the Seaborn style to "white" (no background color)
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
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(x_values, y_values, label='Sampled function', color='b', lw=2)
    
    plt.scatter(training_x, training_y, color='r', label='Training samples', zorder=5)
    plt.scatter(test_x, test_y, color='g', label='Test samples', zorder=5)
    
    plt.title('Sampled function with Training and Test Samples')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    plt.legend()
    
    plt.show()

    print("Training and test datasets (x and y values) have been saved as Numpy arrays.")