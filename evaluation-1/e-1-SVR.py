import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# For handling non-relevant warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # Experimental description
    description = "e-1-SVR"
    timestamp = datetime.now()
    
    # Evaluation parameters
    experimental_runs = 10000 # Max 10 000 runs, otherwise the subsitute dataset will run out
    max_queries = 100

    # Model trackning
    tracker_effectiveness_all = np.zeros((experimental_runs, max_queries), dtype=float)
    tracker_effectiveness_average = np.zeros(max_queries, dtype=float)

    # Loading substitute dataset (prepared)
    dataset_substitute = np.load('evaluation-1/datasets/substitute-dataset/all_experiments.npy', allow_pickle=True)
    # print(dataset_substitute.shape) # Debugging

    # Load the test set (previously saved as 'test.npy')
    dataset_test = np.load('evaluation-1/datasets/target-dataset/test_data.npy', allow_pickle=True)
    test_x = dataset_test[0,:].reshape(-1, 1)
    test_y = dataset_test[1,:]

    # Initialize PolynomialFeatures to transform the input data to the specified degree
    # poly = PolynomialFeatures(degree=degree)

    for e in tqdm(range(experimental_runs), desc ="Running experiment: "):
        # Code that needs to be restarted for each experiment

        e_train_x = dataset_substitute[e][:, 0]
        e_train_y = dataset_substitute[e][:, 1]

        for q in range(1, max_queries + 1):
            q_train_x = dataset_substitute[e][:q, 0].reshape(-1, 1)
            q_train_y = dataset_substitute[e][:q, 1]

            # print(f"q: {q} - {q_train_x.shape} - {q_train_y.shape}")
            # print(f"q: {q} - {len(q_train_x)} - {len(q_train_y)}")

            # Reinitialize the Ridge regression model for each iteration
            model = SVR(kernel='rbf', C=100.0, epsilon=0.001, gamma=1)

            # Train the Ridge regression model with the current set of data
            model.fit(q_train_x, q_train_y)

            # Predict on the test set (transform test inputs similarly)
            y_pred = model.predict(test_x)

            rmse = np.sqrt(mean_squared_error(test_y, y_pred))
            
            tracker_effectiveness_all[e,q-1] = rmse # NOTE: Stored with base index 0,0

    # Averaging all efficiency metrics
    tracker_effectiveness_average = np.mean(tracker_effectiveness_all, axis=0).reshape(1, -1)

    print(tracker_effectiveness_average)
        
    with open('evaluation-1/results/e-1-SVR/tracker_effectiveness_all.pickle', 
                'wb') as file:
        pickle.dump(tracker_effectiveness_all, file)

    with open('evaluation-1/results/e-1-SVR/tracker_effectiveness_average.pickle', 
                'wb') as file:
        pickle.dump(tracker_effectiveness_average, file)
