"""model_analysis.py

this script is designed to analyse models previously trained with model.py
As we are using classification, auc and accuracy are good indicators
of efficiency.
A part of the study could be on the best parameters for our models and wich one is better, we will not do it here
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from timeit import default_timer as timer
from joblib import dump, load

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score
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
cv                = []


""" Auto Label

function used to labelise bar in bar chart
"""
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


""" Data Loading

Load our data for analysis, then fit them in a PCA.
"""
X_train = np.loadtxt(data_path + 'Train/X_train.txt', delimiter=' ')
Y_train = np.loadtxt(data_path + 'Train/y_train.txt')
X_test  = np.loadtxt(data_path + 'Test/X_test.txt', delimiter=' ')
Y_test  = np.loadtxt(data_path + 'Test/y_test.txt')

pca     = PCA(n_components = 7) # 7 components was used for training
X_test  = pca.fit_transform(X_test)


""" Models Loading

Load our models for analysis.
"""
for i in range(3) :
    cv.append(load(save_path_model + "models_" + str(i) + ".pkl"))


""" model analysis

Calculate the AUC and accuracy to compare the models.
"""
models        = ["LogisticRegression", "RandomForest", "SVM"]
x             = np.arange(len(models))  # the label locations
width         = 0.35  # the width of the bars

logistic      = np.zeros((2))
random_forest = np.zeros((2))
svc           = np.zeros((2))

for i in range(3) :
    Y_score = np.array(cv[i].predict_proba(X_test))

    if i == 0:
        logistic[0]      = roc_auc_score(Y_test, Y_score, multi_class = 'ovo')
        logistic[1]      = accuracy_score(Y_test, Y_score.argmax(axis=1))

    if i == 1:
        random_forest[0] = roc_auc_score(Y_test, Y_score, multi_class = 'ovo')
        random_forest[1] = accuracy_score(Y_test, Y_score.argmax(axis=1))

    if i == 2:
        svc[0]           = roc_auc_score(Y_test, Y_score, multi_class = 'ovo')
        svc[1]           = accuracy_score(Y_test, Y_score.argmax(axis=1))

roc      = np.array([logistic[0], random_forest[0], svc[0]], dtype = np.float16)
accuracy = np.array([logistic[1], random_forest[1], svc[1]], dtype = np.float16)

fig, ax  = plt.subplots()
rects1   = ax.bar(x - width/2, roc, width, label='AUC')
rects2   = ax.bar(x + width/2, accuracy, width, label='Accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('%')
ax.set_title('Scores for models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
autolabel(rects1)
autolabel(rects2)
plt.savefig( os.path.join( save_path_graph, f'models_auc_accuracy.png' ) )
plt.clf()

compute_time = timer() - start
print("\nTime: " + str(compute_time))
