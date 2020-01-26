import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

from timeit import default_timer as timer

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC


data_path = "HAPT Data Set/"
save_path = "graph/"

start     = timer()

""" Data Loading

Load our data for analysis
"""
X_train = np.loadtxt(data_path + 'Train/X_train.txt', delimiter=' ')
Y_train = np.loadtxt(data_path + 'Train/y_train.txt')
X_test  = np.loadtxt(data_path + 'Test/X_test.txt', delimiter=' ')
Y_test  = np.loadtxt(data_path + 'Test/y_test.txt')


""" Data Loading

Load our data for analysis
"""
# max_features = list(range(2,int(len(X[0,:]))))
# params = [{"C":[0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000], "penalty":("l1", "l2"), "max_iter":[1000, 10000, 100000]}, {"n_estimators":[50, 100, 200], "max_depth":[2, 3, 5], "min_samples_leaf":[0.1, 0.2, 0.3], "max_features":max_features}, {"C":[0.001, 0.01, 0.1, 1, 10, 100], "max_iter":[1000, 10000, 100000, 1000000]}]
# models = [LogisticRegression(random_state = 42, solver = "liblinear"), RandomForestClassifier(random_state = 42), SVC(random_state = 42, kernel = "linear", probability = True)]
# kf = KFold(len(X))
# predict_bag = np.zeros((len(models), len(X), 2), dtype = np.float32)
# best_param = [[] for i in range(len(models))]
#
# for train, test in kf.split(X):
#     for i, (model, param) in enumerate(zip(models, params)):
#         best_m = GridSearchCV(model, param, scoring=scoring, refit='AUC', cv = 5, n_jobs = 6)
# #            best_m = GridSearchCV(model, param, cv = 5, n_jobs = 8)
#         best_m.fit(X[train], y[train])
#         best_param[i].append(best_m.best_params_)
#         predict_bag[i, test] = best_m.predict_proba(X[test])
#     print("Done " + str(test) + "/" + str(len(X)-1))


compute_time = timer() - start
print("Time: " + str(compute_time))
