"""data_handle.py

this script is designed to select revelant features from the dataset and create
a simplified sample of data for classification model training purpose
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from timeit import default_timer as timer
from sklearn.decomposition import PCA
from joblib import dump, load
from scipy import stats

DATA_PATH = "HAPT Data Set"

start = timer()

""" Data Loading

Load our data for analysis
"""
X_train = np.loadtxt(DATA_PATH + '/Train/X_train.txt', delimiter=' ')
Y_train = np.loadtxt(DATA_PATH + '/Train/y_train.txt')
X_test = np.loadtxt(DATA_PATH + '/Test/X_test.txt', delimiter=' ')
Y_test = np.loadtxt(DATA_PATH + '/Test/y_test.txt')
# train_size = int(X_tr.shape[0]*0.8)
# X_train = X_tr[0:train_size]
# Y_train = Y_tr[0:train_size]
# X_val = X_tr[train_size:]
# Y_val = Y_tr[train_size:]


def DisplayCorr(data, seg):

    data_pd = pd.DataFrame(data)
    corr_matrix = data_pd.corr()


#    sns.heatmap(corr_matrix)
    corr = np.zeros((corr_matrix.shape[0]))

    for i in range(len(data[0,:])):
        temp = 0
        for j in range(corr_matrix.shape[0]):
            if abs(corr_matrix.iat[i,j]) > 0.75:
                temp +=1
        corr[i] = temp


    plt.bar(range(len(corr)), corr)
    plt.xlabel("decriptor ID")
    plt.ylabel("nb of correlation")
    plt.title("correlation for " + seg)
    plt.show()

#    print(stats.ttest_1samp(data_pd,0.0))

#    print(corr_matrix[40].sort_values(ascending = False))

"""PCA

Apply a PCA to know how much explanation we can have with 4 super-parameters
"""
# pca = PCA(n_components = 0.95)
#     pca.fit_transform(X)
# #    print(pca.explained_variance_ratio_)
# #    print(pca.components_.T[:,0])
#     mean = np.zeros((len(pca.components_[0,:])))
#     for i in range(len(pca.components_[0,:])):
#         for j in range(len(pca.components_[:,0])):
#             mean[i] += pca.components_[j,i] * pca.explained_variance_ratio_[j]
#         mean[i] = mean[i] / len(pca.components_[:,0])
#    print(mean)
#    print(pca.singular_values_)

# pca_mean = np.zeros((4, SIZE))
# pca_mean[0, :] = PrincipalComponentAnalysis(X_EST)
# pca_mean[1, :] = PrincipalComponentAnalysis(X_ACT)
# pca_mean[2, :] = PrincipalComponentAnalysis(X_AUTO_15)
# pca_mean[3, :] = PrincipalComponentAnalysis(X_AUTO_20)
#
# mean = np.zeros((SIZE))
# for i in range(SIZE):
#     mean[i] = np.mean(abs(pca_mean[:, i]))
#
# mean_sum = mean.sum()
# for i in range(SIZE):
#     mean[i] = (mean[i]/mean_sum)*100
#
# plt.bar(range(1,len(mean)+1), mean, align='center', alpha=0.5)
# plt.xlabel("Descriptors")
# plt.ylabel('Explanation')
# plt.title('PCA explanation by descriptors')
#
# plt.show()

compute_time = timer() - start
