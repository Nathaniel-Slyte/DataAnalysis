"""data_handle.py

this script is designed to select revelant features from the dataset and create
a simplified sample of data for classification model training purpose
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from timeit import default_timer as timer
from sklearn.decomposition import PCA
from joblib import dump, load
from scipy import stats

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


"""PCA and visualization

Apply a PCA and make differents graph of it
"""
pca = PCA(n_components = 0.95)
pca.fit_transform(X_train)

plt.bar(np.arange(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, align='center', alpha=0.5)
plt.xlabel("Components")
plt.ylabel("Explanation")
plt.title("Explanation by components for 0.95 variance explained PCA")
plt.savefig( os.path.join( save_path, f'PCA_95.png' ) )
plt.clf()

maj_bar       = []
nb_components = 0
while nb_components < len(pca.explained_variance_ratio_) :
    if pca.explained_variance_ratio_[nb_components] >= 0.01 :
        maj_bar.append(pca.explained_variance_ratio_[nb_components])
    nb_components += 1

print(maj_bar)

plt.bar(np.arange(1,len(maj_bar)+1), maj_bar, align='center', alpha=0.5, label=np.sum(maj_bar))
plt.xlabel("Components")
plt.ylabel("Explanation")
plt.title("Components with more than 0.01 variance explained")
plt.legend()
plt.savefig( os.path.join( save_path, f'PCA_95_sup_01.png' ) )
plt.clf()

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Components")
plt.ylabel("Explanation")
plt.title("Evolution of variance explained")
plt.savefig( os.path.join( save_path, f'PCA_95_evolution.png' ) )
plt.clf()


compute_time = timer() - start
print("Time: " + str(compute_time))
