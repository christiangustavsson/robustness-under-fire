# Guide

- /datasets: contains the target, test, surrogate, and substitute datasets for this work. Code for performing several tasks related to curating the CIFAR-10 dataset; however, using the existing sets is advisable. **Note:** The folder contents are large.

- /models/target-model: contains the target model involved in the work. This model can only be loaded from a package file. The folder also contains logs from running hyperparameter optimization for the target model.

- building-substitute-dataset.py: Builds the substitute dataset by simulating an attack against the target model. 

- hyperparameter-optimizing-candidates.py: Tests a set of candidate models over a range of hyperparameters. Note: This can take a -very- long time to run. Saves logs for each of the trials.

- /results: Contain the logs from hyperparameter optimization for all the substitute model candidates. The best results are plotted by e-2-plot-results.py

- e-2-plot-results.py: loads and plots the results.
