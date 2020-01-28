"""model_analysis.py

this script is designed to analyse models previously trained with model.py
As we are using classification, roc curves and confusion matrix are good indicators
of efficiency.
A part of the study is on the best parameters for our models and wich one is better
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from timeit import default_timer as timer
from joblib import dump, load

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


kfold_size        = 5
data_path         = "HAPT Data Set/"
save_path_model   = "saves/"
save_path_graph   = "graph/"

start             = timer()


""" Data Loading

Load our data for analysis, then fit them in a PCA.
"""
X_train = np.loadtxt(data_path + 'Train/X_train.txt', delimiter=' ')
Y_train = np.loadtxt(data_path + 'Train/y_train.txt')
X_test  = np.loadtxt(data_path + 'Test/X_test.txt', delimiter=' ')
Y_test  = np.loadtxt(data_path + 'Test/y_test.txt')

pca     = PCA(n_components = 0.75)
X_test  = pca.fit_transform(X_test)


""" Models Loading

Load our models for analysis.
"""
# cv = load(save_path_model + "models_" + str(i) + "_epoch_" + str(counter + 1) + ".pkl")
