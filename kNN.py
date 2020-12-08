# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:31:21 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
matrix = np.loadtxt('param.txt', dtype = 'f')
embd = np.zeros( (2048, 5) )
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
target = np.loadtxt('data.txt', dtype = 'f')
for i in range(2048):        #labelling the dataset
    if target[i] > 0:
        target[i]=1
    else:
        target[i]=0    

X_train , X_test, y_train,  y_test = train_test_split(matrix, target, random_state=44, train_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10) 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
