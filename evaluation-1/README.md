# Guide

- /datasets/target-dataset: target and test datasets from the sampled function.

- /datasets/substitute-dataset: substitute dataset built by simulating an attack against the target model. A new dataset can be built by running target-model-theft.py.

- /models/target-model: contains the target model involved in the work. This model can only be loaded from a package file.

- /models/optimizing: contain the files that find the optimal properties for each architecture. The best parameters are then used in e-1-GPR.py, e-1-RR.py, and e-1-SVR.py.

- e-1-GPR.py, e-1-RR.py, and e-1-SVR.py: train a substitute model for the three architectures involved in the evaluation. The results are stored in /results.

- e-1-plot-results.py: loads and plots data from /results.
