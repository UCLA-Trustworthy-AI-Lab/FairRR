import numpy as np
import pandas as pd
from sklearn import metrics
import random
import torch
import time
import pickle
# Synthetic Data Vault
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE

# sklearn package
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Other packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


import random

###normalize training and test data###
def normalize(X_train,X_test):
    scaler_X = StandardScaler()

    X_train_ = scaler_X.fit_transform(X_train)
    X_test_ = scaler_X.fit_transform(X_test)

    return X_train_, X_test_



###calculate flipping probabilities under given t###
def Find_Theta(t,p11,p10,p01,p00,fairness):
    if t >= 0:
        if fairness == 'DP':
            theta11 = t / (p11 + p10 + t)
            theta00 = t / (p01 + p00 + t)
            theta10 = 0
            theta01 = 0
        if fairness == 'EO':
            theta11 = t / 2 / p11
            theta00 = t / 2 / (p01 + t)
            theta10 = 0
            theta01 = 0
        if fairness == 'PE':
            theta11 = t / 2 / (p10 + t)
            theta00 = t / 2 / p00
            theta10 = 0
            theta01 = 0
    if t < 0:
        if fairness == 'DP':
            theta11 = 0
            theta00 = 0
            theta10 = t / (t - p11 - p10)
            theta01 = t / (t - p01 - p00)
        if fairness == 'EO':
            theta11 = 0
            theta00 = 0
            theta10 = t / 2 / (t - p11)
            theta01 = -1 * t / 2 / p01
        if fairness == 'PE':
            theta11 = 0
            theta00 = 0
            theta10 = -1 * t / 2 / p10
            theta01 = t / 2 / (t - p00)

    return theta11, theta10, theta01, theta00



#
#
# def csss(Y_train,Z_train,num_sample,t):
#     p11 = ((Z_train==1) & (Y_train==1)).mean()
#     p10 = ((Z_train==1) & (Y_train==0)).mean()
#     p01 = ((Z_train==0) & (Y_train==1)).mean()
#     p00 = ((Z_train==0) & (Y_train==0)).mean()
#     s11 = p11 * (1 - 2 * t * (p01 + p00))
#     s10 = p10 * (1 + 2 * t * (p01 + p00))
#     s01 = p01 * (1 + 2 * t * (p11 + p10))
#     s00 = p00 * (1 - 2 * t * (p11 + p10))
#     s_sum = s11+s10+s01+s00
#     s11,s10,s01,s00 = s11/s_sum, s10/s_sum, s01/s_sum, s00/s_sum
#     s11_count = round(num_sample * s11)
#     s10_count = round(num_sample * s10)
#     s01_count = round(num_sample * s01)
#     s00_count = round(num_sample * s00)
#
#     return s11_count,s10_count,s01_count,s00_count



#
#
#
# def csss1(Y_train,Z_train):
#     n = len(Y_train)
#     n11 = ((Z_train==1) & (Y_train==1)).sum()
#     n10 = ((Z_train==1) & (Y_train==0)).sum()
#     n01 = ((Z_train==0) & (Y_train==1)).sum()
#     n00 = ((Z_train==0) & (Y_train==0)).sum()
#     s11 = (n11+n10) * (n11+n01) / n
#     s10 = (n11+n10) * (n10+n00) /n
#     s01 = (n01+n00) * (n11+n01) /n
#     s00 = (n01+n00) * (n10+n00) /n
#
#
#     s11_count = round(  s11)
#     s10_count = round(  s10)
#     s01_count = round(  s01)
#     s00_count = round(  s00)
#
#     return s11_count,s10_count,s01_count,s00_count

### sample points through SMOTE
def sample_points(dataset, alldata, number):
    idxs = []
    s_lists = []

    for t in range(number):
        s = -1
        while s not in alldata.index:
            idx = random.choices(dataset.index, weights=dataset['weight'])[0]
            points_to_selects = dataset.loc[idx, :]['neighbour']
            s = int(random.choices(points_to_selects.strip('[').strip(']').split(','))[0])
        idxs.append(idx)
        s_lists.append(s)
    data0 = dataset.loc[idxs, :].drop(['neighbour', 'weight'], axis=1)
    data1 = alldata.loc[s_lists, :].drop(['neighbour', 'weight'], axis=1)
    beta = np.random.random((number, 1))

    new_data = data0.values * (1 - beta) + data1.values * beta

    columns = data0.keys()
    syndataset = pd.DataFrame(new_data, columns=columns)

    return syndataset

### calculate accuracy and disparity level of given fairness meature, given training set and given classifier
def train_test_acc_parity(X_syn, Y_syn, X_test, Y_test, Z_test, fairness, classifier):

    svm_C = classifier
    X_syn, X_test= normalize(X_syn,X_test)


    svm_C.fit(X_syn, Y_syn)
    pred = svm_C.predict(X_test)
    acc = metrics.accuracy_score(Y_test, pred)
    f1 =metrics.f1_score(Y_test, pred)
    if fairness =='DP':
        disparity = pred[Z_test == 1].mean() - pred[Z_test == 0].mean()
    if fairness == 'EO':
        disparity = pred[(Z_test == 1) & (Y_test == 1)].mean() - pred[(Z_test == 0) & (Y_test == 1)].mean()
    if fairness == 'PE':
        disparity = pred[(Z_test == 1) & (Y_test == 0)].mean() - pred[(Z_test == 0) & (Y_test == 0)].mean()


    return acc, f1,disparity


### calculate accuracy and disparity levels of given training set and given classifier
def train_test_acc_all_parity(X_syn, Y_syn, X_test, Y_test, Z_test,  classifier):

    svm_C = classifier
    X_syn, X_test= normalize(X_syn,X_test)


    svm_C.fit(X_syn, Y_syn)
    pred = svm_C.predict(X_test)
    acc = metrics.accuracy_score(Y_test, pred)
    f1 =metrics.f1_score(Y_test, pred)
    DDP = pred[Z_test == 1].mean() - pred[Z_test == 0].mean()
    DEO = pred[(Z_test == 1) & (Y_test == 1)].mean() - pred[(Z_test == 0) & (Y_test == 1)].mean()
    DPE = pred[(Z_test == 1) & (Y_test == 0)].mean() - pred[(Z_test == 0) & (Y_test == 0)].mean()


    return acc, f1,DDP,DEO,DPE
