"""data_handle.py

this script is designed to select revelant features from the dataset and create
a simplified sample of data for classification model training purpose
"""

import os, csv, json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from timeit import default_timer as timer
from sklearn.decomposition import PCA
from joblib import dump, load
from scipy import stats




if __name__ == '__main__' :
