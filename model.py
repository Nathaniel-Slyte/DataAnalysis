"""model.py

this script is designed to train differents models with our data for futur analysis.
We use a nested cross-validation to get extra information for an accurate result.
"""
import numpy as np

from timeit import default_timer as timer
from joblib import dump, load

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.simplefilter("ignore")


kfold_size        = 5
data_path         = "HAPT Data Set/"
save_path_model   = "saves/"
save_path_graph   = "graph/"

start             = timer()

""" Data Loading

Load our data for analysis, then fit them in a PCA
"""
X_train = np.loadtxt(data_path + 'Train/X_train.txt', delimiter=' ')
Y_train = np.loadtxt(data_path + 'Train/y_train.txt')
X_test  = np.loadtxt(data_path + 'Test/X_test.txt', delimiter=' ')
Y_test  = np.loadtxt(data_path + 'Test/y_test.txt')

pca     = PCA(n_components = 0.75)
X_train = pca.fit_transform(X_train)


""" Nested Cross-validation

Do a nested cross-validation, with a cross-validation gridsearch, and save all the models for analysis
"""
params         = [{"C":[0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000], "penalty":("l1", "l2"), "max_iter":[1000, 10000, 100000]}, {"n_estimators":[50, 100, 200], "max_depth":[2, 3, 5], "min_samples_leaf":[0.1, 0.2, 0.3]}, {"C":[0.001, 0.01, 0.1, 1, 10, 100], "max_iter":[1000, 10000, 100000, 1000000]}]
models         = [LogisticRegression(random_state = 42, solver = "liblinear"), RandomForestClassifier(random_state = 42), SVC(random_state = 42, kernel = "linear", probability = True)]
kf             = KFold(kfold_size)
counter        = 0

for train, validate in kf.split(X_train):
    for i, (model, param) in enumerate(zip(models, params)):
        best_m = GridSearchCV(model, param, cv = 5, n_jobs = 6)
        best_m.fit(X_train[train], Y_train[train])

        dump(params, save_path_model + "models_" + str(i) + "_epoch_" + str(counter + 1) + ".pkl")
    counter    = counter + 1
    print("Done " + str(counter) + "/" + str(kfold_size))


compute_time = timer() - start
print("\nTime: " + str(compute_time))
